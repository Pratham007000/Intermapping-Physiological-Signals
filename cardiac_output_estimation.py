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
warnings.filterwarnings('ignore')

# 1. Advanced Preprocessing
def denoise_signal(signal_data, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs, wavelet)
    return signal.medfilt(denoised, kernel_size=5)

def signal_quality_check(signal_data, threshold=0.1):
    if np.std(signal_data) < threshold or np.any(~np.isfinite(signal_data)):
        return False
    freqs, psd = signal.welch(signal_data, fs=125, nperseg=min(len(signal_data), 500))
    high_freq_power = np.sum(psd[freqs > 1.0]) / np.sum(psd)
    if high_freq_power > 0.5:
        return False
    return True

def extract_pulse_transit_time(ecg, ppg, fs):
    if not signal_quality_check(ecg) or not signal_quality_check(ppg):
        return 0
    
    ecg = denoise_signal(ecg)
    ppg = denoise_signal(ppg)
    
    # Detect R-peaks in ECG
    ecg_peaks, _ = signal.find_peaks(ecg, distance=int(fs*0.4), prominence=0.5)
    # Detect peaks in PPG
    ppg_peaks, _ = signal.find_peaks(ppg, distance=int(fs*0.4), prominence=0.5)
    
    # Match closest PPG peak after each ECG peak
    ptt_values = []
    for ecg_peak in ecg_peaks:
        valid_ppg_peaks = ppg_peaks[ppg_peaks > ecg_peaks]
        if len(valid_ppg_peaks) > 0:
            ptt = (valid_ppg_peaks[0] - ecg_peak) / fs
            if 0.1 <= ptt <= 0.5:  # Reasonable PTT range (100-500 ms)
                ptt_values.append(ptt)
    
    return np.mean(ptt_values) if ptt_values else 0

# Extract Features for CO
def extract_co_features(ecg, ppg, fs):
    # Heart Rate from ECG
    ecg_peaks, _ = signal.find_peaks(ecg, distance=int(fs*0.4), prominence=0.5)
    hr = 60 / (np.mean(np.diff(ecg_peaks) / fs)) if len(ecg_peaks) >= 2 else 60
    
    # Pulse Transit Time
    ptt = extract_pulse_transit_time(ecg, ppg, fs)
    
    # PPG amplitude as proxy for stroke volume
    ppg_peaks, _ = signal.find_peaks(ppg, distance=int(fs*0.4), prominence=0.5)
    sv_proxy = np.mean(ppg[ppg_peaks]) if len(ppg_peaks) >= 2 else 0
    
    # Frequency-domain feature
    freqs, psd = signal.welch(ppg, fs=fs, nperseg=min(len(ppg), fs*4))
    low_freq_band = (freqs >= 0.05) & (freqs <= 0.15)  # LF band for vascular tone
    lf_power = np.sum(psd[low_freq_band]) if np.any(low_freq_band) else 0
    
    return np.array([hr, ptt, sv_proxy, lf_power])

# Simulate CO ground truth (since BIDMC doesn't have CO)
def simulate_co_ground_truth(ecg, fs, hr):
    # Simplified CO = HR * SV, with SV estimated from ECG amplitude
    ecg_peaks, _ = signal.find_peaks(ecg, distance=int(fs*0.4), prominence=0.5)
    sv = np.mean(ecg[ecg_peaks]) * 70 if len(ecg_peaks) >= 2 else 70  # Arbitrary scaling
    co = (hr / 60) * sv  # CO in L/min
    return max(2.0, min(8.0, co))  # Reasonable CO range

# Load and preprocess data
def load_and_preprocess_data(window_size=500):
    records = ['bidmc01', 'bidmc02', 'bidmc03', 'bidmc04', 'bidmc05']
    X_all, F_all, Y_all = [], [], []
    
    for record_name in records:
        try:
            print(f"Loading record: {record_name}")
            record = wfdb.rdrecord(record_name, pn_dir='bidmc/1.0.0/')
            fs = record.fs
            ecg = record.p_signal[:, 0]
            ppg = record.p_signal[:, 1]
            
            # Normalize signals
            ecg = (ecg - np.mean(ecg)) / np.std(ecg)
            ppg = (ppg - np.mean(ppg)) / np.std(ppg)
            
            # Create overlapping windows
            step_size = window_size // 2
            num_samples = (len(ecg) - window_size) // step_size + 1
            X = np.zeros((num_samples, window_size, 2))
            F = np.zeros((num_samples, 4))  # 4 features
            Y = np.zeros(num_samples)
            
            for i in range(num_samples):
                start = i * step_size
                end = start + window_size
                X[i, :, 0] = ecg[start:end]
                X[i, :, 1] = ppg[start:end]
                features = extract_co_features(ecg[start:end], ppg[start:end], fs)
                F[i] = features
                # Simulate CO ground truth
                hr = features[0]
                Y[i] = simulate_co_ground_truth(ecg[start:end], fs, hr)
            
            valid_idx = Y > 0
            if np.sum(valid_idx) > 0:
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
    def __init__(self, input_size, d_model, nhead, num_transformer_layers, num_cnn_filters=32, dropout=0.3):
        super(HybridTransformerCNN, self).__init__()
        self.d_model = d_model
        
        # CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, num_cnn_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(num_cnn_filters),
            nn.Conv1d(num_cnn_filters, num_cnn_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(num_cnn_filters),
        )
        
        self.input_projection = nn.Linear(num_cnn_filters, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                 dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model + 4, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        self.residual = nn.Linear(d_model + 4, 1)

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
    weights = 1.0 + (mae / 2.0)
    return torch.mean(mae * weights)

# 4. Training Loop
def train_model(model, train_loader, val_loader, device, num_epochs=100, patience=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    os.makedirs('co_model_checkpoints', exist_ok=True)
    
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
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val CO MAE: {val_mae:.2f} L/min')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'co_model_checkpoints/best_co_hybrid_model.pth')
            print(f'Model saved at epoch {epoch+1} with validation loss: {val_loss:.6f}')
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
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, F_batch, y_batch in test_loader:
            X_batch, F_batch = X_batch.to(device), F_batch.to(device)
            pred = model(X_batch, F_batch).squeeze()
            y_true.extend(y_batch.numpy())
            y_pred.extend(pred.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    
    avg_actual_co = np.mean(y_true)
    avg_predicted_co = np.mean(y_pred)
    print(f'\n=== Test Results ===')
    print(f'Test CO MAE: {mae:.2f} L/min')
    print(f'Average Actual CO: {avg_actual_co:.2f} L/min')
    print(f'Average Predicted CO: {avg_predicted_co:.2f} L/min')
    print(f'=== End of Test Results ===\n')
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(y_true[:100], label='Actual CO (L/min)', color='green')
    plt.plot(y_pred[:100], label='Predicted CO (L/min)', color='red', linestyle='--')
    plt.title('Actual vs Predicted CO (First 100 Sequences)')
    plt.xlabel('Sequence Index')
    plt.ylabel('CO (L/min)')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f'Predicted vs Actual CO (MAE={mae:.2f} L/min)')
    plt.xlabel('Actual CO (L/min)')
    plt.ylabel('Predicted CO (L/min)')
    
    plt.subplot(3, 1, 3)
    errors = y_pred - y_true
    plt.hist(errors, bins=50, color='blue', alpha=0.7)
    plt.title('Error Distribution (Predicted - Actual CO)')
    plt.xlabel('Error (L/min)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return mae, avg_actual_co, avg_predicted_co

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
    
    X, F, Y = load_and_preprocess_data(window_size=500)
    
    print(f'Memory after data loading: {get_memory_usage():.1f} MB')
    cleanup_memory()
    
    print(f'Data shapes: X={X.shape}, F={F.shape}, Y={Y.shape}')
    if X.shape[0] == 0 or Y.shape[0] == 0:
        print("Error: No valid data extracted!")
        return
    print(f'X range: [{X.min():.3f}, {X.max():.3f}]')
    print(f'Y range (CO in L/min): [{Y.min():.3f}, {Y.max():.3f}]')
    print(f'Y mean (CO in L/min): {Y.mean():.3f}, Y std: {Y.std():.3f}')
    
    X_temp, X_test, F_temp, F_test, Y_temp, Y_test = train_test_split(X, F, Y, test_size=0.15, random_state=42)
    X_train, X_val, F_train, F_val, Y_train, Y_val = train_test_split(X_temp, F_temp, Y_temp, test_size=0.15/0.85, random_state=42)
    print(f'Dataset sizes: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}')
    
    X_train = torch.FloatTensor(X_train)
    F_train = torch.FloatTensor(F_train)
    Y_train = torch.FloatTensor(Y_train)
    X_val = torch.FloatTensor(X_val)
    F_val = torch.FloatTensor(F_val)
    Y_val = torch.FloatTensor(Y_val)
    X_test = torch.FloatTensor(X_test)
    F_test = torch.FloatTensor(F_test)
    Y_test = torch.FloatTensor(Y_test)
    
    print(f'Memory after tensor conversion: {get_memory_usage():.1f} MB')
    cleanup_memory()
    
    batch_size = 8
    train_dataset = TensorDataset(X_train, F_train, Y_train)
    val_dataset = TensorDataset(X_val, F_val, Y_val)
    test_dataset = TensorDataset(X_test, F_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f'Using batch size: {batch_size}')
    print(f'Memory after DataLoader creation: {get_memory_usage():.1f} MB')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridTransformerCNN(
        input_size=2, 
        d_model=64, 
        nhead=4, 
        num_transformer_layers=2, 
        num_cnn_filters=32, 
        dropout=0.3
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Memory after model creation: {get_memory_usage():.1f} MB')
    
    print('Starting training...')
    start_time = time.time()
    train_model(model, train_loader, val_loader, device, num_epochs=100, patience=30)
    print(f'Training completed in {time.time() - start_time:.2f} seconds')
    
    print('Loading best model from co_model_checkpoints/best_co_hybrid_model.pth')
    model.load_state_dict(torch.load('co_model_checkpoints/best_co_hybrid_model.pth'))
    
    print('\nEvaluating on test set...')
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()