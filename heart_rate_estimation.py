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
    try:
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        denoised = pywt.waverec(coeffs, wavelet)
        return signal.medfilt(denoised, kernel_size=3)
    except ValueError as e:
        print(f"Error in denoising: {e}. Returning original signal.")
        return signal.medfilt(signal_data, kernel_size=3)

def signal_quality_check(signal_data, threshold=0.05):
    if len(signal_data) < 125 or np.std(signal_data) < threshold or np.any(~np.isfinite(signal_data)):
        return False
    freqs, psd = signal.welch(signal_data, fs=125, nperseg=min(len(signal_data), 500))
    high_freq_power = np.sum(psd[freqs > 10.0]) / np.sum(psd)
    return high_freq_power <= 0.4

def extract_heart_rate_ecg(ecg_signal, fs, min_distance=0.3, prominence=None):
    if len(ecg_signal) < fs or not signal_quality_check(ecg_signal):
        return 0
    
    ecg_signal = denoise_signal(ecg_signal)
    
    sos = signal.butter(4, [1, 40], btype='band', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, ecg_signal)
    
    filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    
    if prominence is None:
        prominence = np.std(filtered) * 0.8
    
    peaks, _ = signal.find_peaks(filtered, 
                                distance=int(min_distance * fs), 
                                prominence=prominence,
                                width=fs*0.05)
    
    if len(peaks) < 2:
        freqs, psd = signal.welch(filtered, fs=fs, nperseg=min(len(filtered), fs*4))
        heart_rate_band = (freqs >= 0.5) & (freqs <= 4.0)
        if np.any(heart_rate_band):
            peak_freq = freqs[heart_rate_band][np.argmax(psd[heart_rate_band])]
            hr = peak_freq * 60
            return max(30, min(240, hr))
        return 0
    
    intervals = np.diff(peaks) / fs
    valid_intervals = intervals[(intervals >= 0.25) & (intervals <= 2.0)]
    
    if len(valid_intervals) == 0:
        return 0
    
    mean_interval = np.mean(valid_intervals)
    hr = 60 / mean_interval
    return max(30, min(240, hr))

def extract_features(ecg, ppg, fs):
    ecg_peaks, _ = signal.find_peaks(ecg, distance=int(fs*0.3), prominence=0.5)
    hrv_feature = np.std(np.diff(ecg_peaks) / fs) if len(ecg_peaks) >= 2 else 0
    
    ppg_peaks, _ = signal.find_peaks(ppg, distance=int(fs*0.3), prominence=0.5)
    amp_feature = np.std(ppg[ppg_peaks]) if len(ppg_peaks) >= 2 else 0
    
    freqs, psd = signal.welch(ppg, fs=fs, nperseg=min(len(ppg), fs*4))
    heart_rate_band = (freqs >= 0.5) & (freqs <= 4.0)
    freq_feature = freqs[heart_rate_band][np.argmax(psd[heart_rate_band])] * 60 if np.any(heart_rate_band) else 0
    
    try:
        scales = np.arange(1, 16)
        coeffs, _ = pywt.cwt(ppg[:min(len(ppg), 500)], scales, 'morl', sampling_period=1/fs)
        if np.any(heart_rate_band):
            band_indices = np.where(heart_rate_band)[0]
            if len(band_indices) > 0:
                start_idx = max(0, band_indices[0])
                end_idx = min(coeffs.shape[1], band_indices[-1] + 1)
                cwt_feature = np.std(coeffs[:, start_idx:end_idx]) if start_idx < end_idx else 0
            else:
                cwt_feature = 0
        else:
            cwt_feature = 0
    except:
        cwt_feature = 0
    
    return np.array([hrv_feature, amp_feature, freq_feature, cwt_feature])

def load_and_preprocess_data(window_size=500, data_dir='bidmc/1.0.0/'):
    # Download dataset if not present
    if not os.path.exists(data_dir):
        print(f"Dataset directory {data_dir} not found. Downloading BIDMC dataset...")
        try:
            os.makedirs(data_dir, exist_ok=True)
            wfdb.dl_database('bidmc', data_dir)
            print(f"BIDMC dataset downloaded to {data_dir}")
        except Exception as e:
            print(f"Failed to download BIDMC dataset: {e}")
            print("Please download the dataset manually from https://physionet.org/content/bidmc/1.0.0/ and place it in the correct directory.")
            raise FileNotFoundError(f"Dataset directory {data_dir} not found.")

    records = ['bidmc01', 'bidmc02', 'bidmc03', 'bidmc04', 'bidmc05']
    X_all, F_all, Y_all = [], [], []
    
    for record_name in records:
        try:
            print(f"Loading record: {record_name}")
            record = wfdb.rdrecord(record_name, pn_dir=data_dir)
            fs = record.fs
            if record.p_signal.shape[1] < 2:
                print(f"Skipping {record_name}: Insufficient signal channels.")
                continue
            ecg = record.p_signal[:, 0]
            ppg = record.p_signal[:, 1]
            
            ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
            ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
            
            step_size = window_size // 2
            num_samples = (len(ecg) - window_size) // step_size + 1
            X = np.zeros((num_samples, window_size, 2))
            F = np.zeros((num_samples, 4))
            Y = np.zeros(num_samples)
            
            for i in range(num_samples):
                start = i * step_size
                end = start + window_size
                if end > len(ecg):
                    continue
                X[i, :, 0] = ecg[start:end]
                X[i, :, 1] = ppg[start:end]
                F[i] = extract_features(ecg[start:end], ppg[start:end], fs)
                Y[i] = extract_heart_rate_ecg(ecg[start:end], fs)
            
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

class HybridTransformerCNN(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_transformer_layers, num_cnn_filters=32, dropout=0.3):
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

def custom_mae_loss(y_pred, y_true):
    mae = torch.abs(y_pred - y_true)
    weights = 1.0 + (mae / 5.0)
    return torch.mean(mae * weights)

def train_model(model, train_loader, val_loader, device, num_epochs=50, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    os.makedirs('hr_model_checkpoints', exist_ok=True)
    
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
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val HR MAE: {val_mae:.2f} bpm')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'hr_model_checkpoints/best_hr_hybrid_model.pth')
            print(f'Model saved at epoch {epoch+1} with validation loss: {val_loss:.6f}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        scheduler.step()
        
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    return best_val_loss

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
    
    avg_actual_hr = np.mean(y_true)
    avg_predicted_hr = np.mean(y_pred)
    print(f'\n=== Test Results ===')
    print(f'Test HR MAE: {mae:.2f} bpm')
    print(f'Average Actual HR (ECG): {avg_actual_hr:.2f} bpm')
    print(f'Average Predicted HR (PPG): {avg_predicted_hr:.2f} bpm')
    print(f'=== End of Test Results ===\n')
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(y_true[:100], label='Actual HR (ECG, bpm)', color='green')
    plt.plot(y_pred[:100], label='Predicted HR (PPG, bpm)', color='red', linestyle='--')
    plt.title('Actual (ECG) vs Predicted (PPG) HR (First 100 Sequences)')
    plt.xlabel('Sequence Index')
    plt.ylabel('HR (bpm)')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f'Predicted (PPG) vs Actual (ECG) HR (MAE={mae:.2f} bpm)')
    plt.xlabel('Actual HR (ECG, bpm)')
    plt.ylabel('Predicted HR (PPG, bpm)')
    
    plt.subplot(3, 1, 3)
    errors = y_pred - y_true
    plt.hist(errors, bins=50, color='blue', alpha=0.7)
    plt.title('Error Distribution (Predicted PPG - Actual ECG HR)')
    plt.xlabel('Error (bpm)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return mae, avg_actual_hr, avg_predicted_hr

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    print(f'Initial memory usage: {get_memory_usage():.1f} MB')
    
    try:
        X, F, Y = load_and_preprocess_data(window_size=500)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    print(f'Memory after data loading: {get_memory_usage():.1f} MB')
    cleanup_memory()
    
    print(f'Data shapes: X={X.shape}, F={F.shape}, Y={Y.shape}')
    if X.shape[0] == 0 or Y.shape[0] == 0:
        print("Error: No valid data extracted!")
        return
    print(f'X range: [{X.min():.3f}, {X.max():.3f}]')
    print(f'Y range (HR in bpm): [{Y.min():.3f}, {Y.max():.3f}]')
    print(f'Y mean (HR in bpm): {Y.mean():.3f}, Y std: {Y.std():.3f}')
    
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
    train_model(model, train_loader, val_loader, device, num_epochs=50, patience=20)
    print(f'Training completed in {time.time() - start_time:.2f} seconds')
    
    print('Loading best model from hr_model_checkpoints/best_hr_hybrid_model.pth')
    try:
        model.load_state_dict(torch.load('hr_model_checkpoints/best_hr_hybrid_model.pth', map_location=device))
    except FileNotFoundError:
        print("Error: Best model checkpoint not found!")
        return
    
    print('\nEvaluating on test set...')
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()