import wfdb
import numpy as np
from scipy import signal
import pywt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import time
import psutil
import gc
import warnings
from scipy.interpolate import interp1d
from scipy.signal import periodogram, butter, sosfilt
warnings.filterwarnings('ignore')

# 1. Advanced Preprocessing
def denoise_signal(signal_data, wavelet='db6', level=4):
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs, wavelet)
    return signal.medfilt(denoised, kernel_size=3)

def signal_quality_check(signal_data, fs, threshold_std=0.1, high_freq_threshold=0.2):
    if np.std(signal_data) < threshold_std or np.any(~np.isfinite(signal_data)):
        return False
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=min(len(signal_data), 500))
    high_freq_power = np.sum(psd[freqs > 30.0]) / np.sum(psd)
    return high_freq_power <= high_freq_threshold

def detect_r_peaks(ecg_signal, fs):
    # Band-pass filter (0.5–40 Hz) for ECG
    sos = butter(4, [0.5, 40], btype='band', fs=fs, output='sos')
    filtered = sosfilt(sos, ecg_signal)
    denoised = denoise_signal(filtered)
    
    # Adaptive peak detection
    mean_signal = np.mean(np.abs(denoised))
    prominence = max(0.1 * mean_signal, 0.2)
    peaks, _ = signal.find_peaks(denoised, distance=int(0.25 * fs), prominence=prominence)
    
    if len(peaks) < 3 or not signal_quality_check(denoised, fs):
        return np.array([])
    return peaks

# HRV Feature Extraction
def extract_hrv_features(ecg_signal, fs):
    peaks = detect_r_peaks(ecg_signal, fs)
    if len(peaks) < 3:
        return np.zeros(5), 0.0
    
    # Calculate RR intervals
    rr_intervals = np.diff(peaks) / fs
    # Remove outliers using IQR
    q25, q75 = np.percentile(rr_intervals, [25, 75])
    iqr = q75 - q25
    valid_rr = rr_intervals[(rr_intervals > (q25 - 1.5 * iqr)) & (rr_intervals < (q75 + 1.5 * iqr))]
    if len(valid_rr) < 2:
        return np.zeros(5), 0.0
    
    # Time-domain features
    rmssd = np.sqrt(np.mean(np.diff(valid_rr)**2)) if len(valid_rr) > 1 else 0.0
    sdnn = np.std(valid_rr) if len(valid_rr) > 1 else 0.0
    
    # Frequency-domain features
    if len(valid_rr) >= 10:
        t_rr = np.cumsum(np.concatenate([[0], valid_rr[:-1]]))
        dt = 0.25  # 4 Hz resampling
        t_new = np.arange(t_rr[0], t_rr[-1], dt)
        if len(t_new) > 10:
            rr_resampled = np.interp(t_new, t_rr[:-1], valid_rr * 1000)  # Convert to ms
            freqs, psd = periodogram(rr_resampled, fs=1/dt)
            lf_band = (freqs >= 0.04) & (freqs <= 0.15)
            hf_band = (freqs >= 0.15) & (freqs <= 0.4)
            lf_power = np.sum(psd[lf_band]) if np.any(lf_band) else 0.0
            hf_power = np.sum(psd[hf_band]) if np.any(hf_band) else 0.0
            lf_hf_ratio = lf_power / (hf_power + 1e-8)
        else:
            lf_power, hf_power, lf_hf_ratio = 0.0, 0.0, 1.0
    else:
        lf_power, hf_power, lf_hf_ratio = 0.0, 0.0, 1.0
    
    # Non-linear feature
    def sample_entropy(data, m=2, r=None):
        if r is None:
            r = 0.2 * np.std(data)
        N = len(data)
        if N <= m + 1:
            return 0.0
        def _phi(m):
            x = np.array([data[i:i+m] for i in range(N-m+1)])
            C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=1)
            return np.sum(C) / (N-m+1)
        return -np.log((_phi(m+1) + 1e-8) / (_phi(m) + 1e-8))
    
    samp_en = sample_entropy(valid_rr) if len(valid_rr) > 2 else 0.0
    
    features = np.array([rmssd, sdnn, lf_power, hf_power, samp_en])
    return features, max(0.001, rmssd)

# Load and preprocess BIDMC dataset
def load_and_preprocess_data(window_size=500):
    records = ['bidmc01', 'bidmc02', 'bidmc03', 'bidmc04', 'bidmc05']
    X_all, F_all, Y_all = [], [], []
    
    for record_name in records:
        try:
            print(f"Loading record: {record_name}")
            record = wfdb.rdrecord(record_name, pn_dir='bidmc/1.0.0/')
            fs = record.fs
            ecg = record.p_signal[:, 0]
            ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)  # Preserve morphology
            
            step_size = window_size // 2
            num_samples = (len(ecg) - window_size) // step_size + 1
            X = np.zeros((num_samples, window_size, 1))
            F = np.zeros((num_samples, 5))
            Y = np.zeros(num_samples)
            
            for i in range(num_samples):
                start, end = i * step_size, i * step_size + window_size
                window_ecg = ecg[start:end]
                X[i, :, 0] = window_ecg
                F[i], Y[i] = extract_hrv_features(window_ecg, fs)
            
            valid_idx = Y > 0.0
            if np.sum(valid_idx) > 10:
                X_all.append(X[valid_idx])
                F_all.append(F[valid_idx])
                Y_all.append(Y[valid_idx])
        except Exception as e:
            print(f"Error loading {record_name}: {e}")
            continue
    
    if not X_all:
        raise Exception("No valid data loaded from any record")
    
    X = np.concatenate(X_all, axis=0)
    F = np.concatenate(F_all, axis=0)
    Y = np.concatenate(Y_all, axis=0)
    return X, F, Y

# 2. Hybrid Transformer-CNN Model
class HybridTransformerCNN(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_transformer_layers, num_cnn_filters=64, dropout=0.2):
        super(HybridTransformerCNN, self).__init__()
        self.d_model = d_model
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, num_cnn_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(num_cnn_filters),
            nn.Conv1d(num_cnn_filters, num_cnn_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(num_cnn_filters),
        )
        
        self.input_projection = nn.Linear(num_cnn_filters, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model + 5, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.ReLU()  # Enforce non-negative output
        )
        self.residual = nn.Linear(d_model + 5, 1)

    def forward(self, x, features):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_projection(x) * np.sqrt(self.d_model)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = torch.cat((x, features), dim=1)
        out = self.fc(x) + self.residual(x)
        return out

# 3. Custom Loss Function
def custom_mae_loss(y_pred, y_true):
    mae = torch.abs(y_pred - y_true)
    weights = 1.0 + (mae / 0.02) + 10.0 * torch.relu(-y_pred)  # Penalize negative predictions
    return torch.mean(mae * weights)

# 4. Training Loop
def train_model(model, train_loader, val_loader, device, num_epochs=100, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    os.makedirs('hrv_model_checkpoints', exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, F_batch, y_batch in train_loader:
            X_batch, F_batch, y_batch = X_batch.to(device), F_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch, F_batch).squeeze()
            loss = custom_mae_loss(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for X_batch, F_batch, y_batch in val_loader:
                X_batch, F_batch, y_batch = X_batch.to(device), F_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch, F_batch).squeeze()
                loss = custom_mae_loss(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_mae += torch.abs(y_pred - y_batch).sum().item()
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val RMSSD MAE: {val_mae:.4f} ms')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'hrv_model_checkpoints/best_hrv_hybrid_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        scheduler.step()
        
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    return best_val_loss

# 5. Evaluation and Visualization
def evaluate_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, F_batch, y_batch in test_loader:
            X_batch, F_batch = X_batch.to(device), F_batch.to(device)
            pred = model(X_batch, F_batch).squeeze()
            y_true.extend(y_batch.numpy())
            y_pred.extend(pred.cpu().numpy())
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    avg_actual_rmssd, avg_predicted_rmssd = np.mean(y_true), np.mean(y_pred)
    print(f'\n=== Test Results ===')
    print(f'Test RMSSD MAE: {mae:.4f} ms')
    print(f'Average Actual RMSSD: {avg_actual_rmssd:.4f} ms')
    print(f'Average Predicted RMSSD: {avg_predicted_rmssd:.4f} ms')
    print(f'=== End of Test Results ===\n')
    
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(y_true[:100], label='Actual RMSSD (ms)', color='blue')
    plt.plot(y_pred[:100], label='Predicted RMSSD (ms)', color='orange', linestyle='--')
    plt.title('Actual vs Predicted RMSSD (First 100 Sequences)')
    plt.xlabel('Sequence Index')
    plt.ylabel('RMSSD (ms)')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f'Predicted vs Actual RMSSD (MAE={mae:.4f} ms)')
    plt.xlabel('Actual RMSSD (ms)')
    plt.ylabel('Predicted RMSSD (ms)')
    
    plt.subplot(3, 1, 3)
    errors = y_pred - y_true
    plt.hist(errors, bins=50, color='purple', alpha=0.7)
    plt.title('Error Distribution (Predicted - Actual RMSSD)')
    plt.xlabel('Error (ms)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    return mae, avg_actual_rmssd, avg_predicted_rmssd

# Main Function
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    print(f'Initial memory usage: {get_memory_usage():.1f} MB')
    window_size = 500
    X, F, Y = load_and_preprocess_data(window_size=window_size)
    print(f'Memory after data loading: {get_memory_usage():.1f} MB')
    cleanup_memory()
    
    print(f'Data shapes: X={X.shape}, F={F.shape}, Y={Y.shape}')
    if X.shape[0] == 0 or Y.shape[0] == 0:
        print("Error: No valid data loaded!")
        return
    print(f'X range: [{X.min():.3f}, {X.max():.3f}]')
    print(f'Y range (RMSSD in ms): [{Y.min():.3f}, {Y.max():.3f}]')
    print(f'Y mean (RMSSD in ms): {Y.mean():.3f}, Y std: {Y.std():.3f}')
    
    X_temp, X_test, F_temp, F_test, Y_temp, Y_test = train_test_split(X, F, Y, test_size=0.15, random_state=42)
    X_train, X_val, F_train, F_val, Y_train, Y_val = train_test_split(X_temp, F_temp, Y_temp, test_size=0.15/0.85, random_state=42)
    print(f'Dataset sizes: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}')
    
    X_train, F_train, Y_train = torch.FloatTensor(X_train), torch.FloatTensor(F_train), torch.FloatTensor(Y_train)
    X_val, F_val, Y_val = torch.FloatTensor(X_val), torch.FloatTensor(F_val), torch.FloatTensor(Y_val)
    X_test, F_test, Y_test = torch.FloatTensor(X_test), torch.FloatTensor(F_test), torch.FloatTensor(Y_test)
    print(f'Memory after tensor conversion: {get_memory_usage():.1f} MB')
    cleanup_memory()
    
    batch_size = 16
    train_loader = DataLoader(TensorDataset(X_train, F_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, F_val, Y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, F_test, Y_test), batch_size=batch_size)
    print(f'Using batch size: {batch_size}')
    print(f'Memory after DataLoader creation: {get_memory_usage():.1f} MB')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridTransformerCNN(input_size=1, d_model=128, nhead=4, num_transformer_layers=3, num_cnn_filters=64).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Memory after model creation: {get_memory_usage():.1f} MB')
    
    print('Starting training...')
    start_time = time.time()
    train_model(model, train_loader, val_loader, device, num_epochs=100, patience=20)
    print(f'Training completed in {time.time() - start_time:.2f} seconds')
    
    print('Loading best model from hrv_model_checkpoints/best_hrv_hybrid_model.pth')
    model.load_state_dict(torch.load('hrv_model_checkpoints/best_hrv_hybrid_model.pth'))
    
    print('\nEvaluating on test set...')
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()