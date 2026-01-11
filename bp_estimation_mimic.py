import scipy.io
import numpy as np
from scipy import signal
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

# 1. Preprocessing (Minimal)
def signal_quality_check(signal_data, threshold=0.05):
    if np.any(~np.isfinite(signal_data)) or np.std(signal_data) < threshold:
        return False
    return True

def extract_pulse_rate(ppg_signal, fs, min_distance=0.5, prominence=0.3):
    if len(ppg_signal) < fs or not signal_quality_check(ppg_signal):
        return 0
    peaks, _ = signal.find_peaks(ppg_signal, distance=int(min_distance * fs), prominence=prominence)
    if len(peaks) < 2:
        freqs, psd = signal.welch(ppg_signal, fs=fs, nperseg=min(len(ppg_signal), fs*2))
        pulse_band = (freqs >= 0.5) & (freqs <= 3.0)
        if np.any(pulse_band):
            peak_freq = freqs[pulse_band][np.argmax(psd[pulse_band])]
            pulse_rate = peak_freq * 60
            return max(40, min(120, pulse_rate))
        return 0
    intervals = np.diff(peaks) / fs
    valid_intervals = intervals[(intervals >= 0.5) & (intervals <= 1.5)]
    if len(valid_intervals) == 0:
        return 0
    mean_interval = np.mean(valid_intervals)
    pulse_rate = 60 / mean_interval
    return max(40, min(120, pulse_rate))

def extract_features(ecg, ppg, fs):
    ecg_peaks, _ = signal.find_peaks(ecg, distance=int(fs*0.2), prominence=0.3)
    hrv_feature = np.std(np.diff(ecg_peaks) / fs) if len(ecg_peaks) >= 2 else 0
    ppg_peaks, _ = signal.find_peaks(ppg, distance=int(fs*0.2), prominence=0.3)
    amp_feature = np.std(ppg[ppg_peaks]) if len(ppg_peaks) > 0 else 0
    freqs, psd = signal.welch(ppg, fs=fs, nperseg=min(len(ppg), fs*2))
    freq_feature = freqs[(freqs >= 0.5) & (freqs <= 3.0)][np.argmax(psd[(freqs >= 0.5) & (freqs <= 3.0)])] * 60 if np.any((freqs >= 0.5) & (freqs <= 3.0)) else 0
    # Pulse Transit Time (PTT)
    ptt_feature = 0
    if len(ecg_peaks) > 0 and len(ppg_peaks) > 0:
        ecg_peaks = ecg_peaks[ecg_peaks < min(len(ecg), len(ppg))]
        ppg_peaks = ppg_peaks[ppg_peaks < min(len(ecg), len(ppg))]
        ptt_values = []
        for ecg_peak in ecg_peaks:
            subsequent_ppg_peaks = ppg_peaks[ppg_peaks > ecg_peak]
            if len(subsequent_ppg_peaks) > 0:
                ptt = (subsequent_ppg_peaks[0] - ecg_peak) / fs
                if 0.1 <= ptt <= 0.5:
                    ptt_values.append(ptt)
        ptt_feature = np.mean(ptt_values) if ptt_values else 0
    # PPG Waveform Rise Time
    rise_time_feature = 0
    if len(ppg_peaks) > 1:
        rise_times = []
        for i in range(len(ppg_peaks)-1):
            start = ppg_peaks[i]
            end = ppg_peaks[i+1]
            if end - start > 5:
                segment = ppg[start:end]
                valley_idx = start + np.argmin(segment)
                if valley_idx < end:
                    rise_time = (end - valley_idx) / fs
                    rise_times.append(rise_time)
        rise_time_feature = np.mean(rise_times) if rise_times else 0
    # PPG Slope
    slope_feature = 0
    if len(ppg_peaks) > 1:
        slopes = []
        for i in range(len(ppg_peaks)-1):
            start = ppg_peaks[i]
            end = ppg_peaks[i+1]
            if end - start > 5:
                segment = ppg[start:end]
                rise = np.max(segment) - np.min(segment)
                duration = (end - start) / fs
                slope = rise / duration if duration > 0 else 0
                slopes.append(slope)
        slope_feature = np.mean(slopes) if slopes else 0
    # ECG R-wave Amplitude
    r_wave_amplitude = 0
    if len(ecg_peaks) > 0:
        r_wave_amplitude = np.mean(ecg[ecg_peaks]) - np.mean(np.delete(ecg, ecg_peaks)) if len(ecg) > len(ecg_peaks) else 0
    return np.array([hrv_feature, amp_feature, freq_feature, ptt_feature, rise_time_feature, slope_feature, r_wave_amplitude])

def process_subject_signals(ppg, abp, ecg, window_size, fs, X_all, F_all, Y_all, file_path):
    if len(ppg) < window_size or len(abp) < window_size or len(ecg) < window_size:
        return 0
    step_size = window_size // 2
    num_samples = (len(ppg) - window_size) // step_size + 1
    X = np.zeros((num_samples, window_size, 2))
    F = np.zeros((num_samples, 7))  # 7 features
    Y = np.zeros((num_samples, 2))
    valid_samples = 0
    for i in range(num_samples):
        start = i * step_size
        end = start + window_size
        if not (signal_quality_check(ppg[start:end]) and signal_quality_check(ecg[start:end])):
            continue
        X[valid_samples, :, 0] = ecg[start:end]
        X[valid_samples, :, 1] = ppg[start:end]
        F[valid_samples] = extract_features(ecg[start:end], ppg[start:end], fs)
        abp_window = abp[start:end]
        peaks, _ = signal.find_peaks(abp_window, distance=int(fs*0.1), prominence=0.2)
        valleys, _ = signal.find_peaks(-abp_window, distance=int(fs*0.1), prominence=0.2)
        if len(peaks) > 0 and len(valleys) > 0:
            sbp = np.mean(abp_window[peaks])
            dbp = np.mean(abp_window[valleys])
            if 50 <= sbp <= 200 and 30 <= dbp <= 150 and sbp > dbp:
                Y[valid_samples] = [sbp, dbp]
                valid_samples += 1
        else:
            continue
    if valid_samples > 0:
        X_all.append(X[:valid_samples])
        F_all.append(F[:valid_samples])
        Y_all.append(Y[:valid_samples])
    return valid_samples

def load_and_preprocess_data(data_files, window_size=250, fs=125):
    X_all, F_all, Y_all = [], [], []
    for file_path in data_files:
        try:
            print(f"Loading file: {file_path}")
            mat = scipy.io.loadmat(file_path)
            data_key = 'p'
            if data_key not in mat:
                print(f"No 'p' key found in {file_path}. Available keys: {list(mat.keys())}")
                continue
            data = mat[data_key]
            print(f"Data shape: {data.shape}, dtype: {data.dtype}")
            if data.dtype == 'object' or len(data.shape) == 3:
                data = data.reshape(-1, data.shape[-2], data.shape[-1]) if len(data.shape) == 3 else data.flatten()
                for subject_data in data:
                    if subject_data.shape[0] >= 3:
                        ppg = subject_data[0].flatten()
                        abp = subject_data[1].flatten()
                        ecg = subject_data[2].flatten()
                        print(f"Subject signals: PPG={len(ppg)}, ABP={len(abp)}, ECG={len(ecg)}")
                        processed_count = process_subject_signals(ppg, abp, ecg, window_size, fs, X_all, F_all, Y_all, file_path)
                        if processed_count > 0:
                            print(f"Processed {processed_count} samples from subject in {file_path}")
            elif len(data.shape) == 2 and data.shape[0] >= 3:
                ppg = data[0].flatten()
                abp = data[1].flatten()
                ecg = data[2].flatten()
                print(f"Signals: PPG={len(ppg)}, ABP={len(abp)}, ECG={len(ecg)}")
                processed_count = process_subject_signals(ppg, abp, ecg, window_size, fs, X_all, F_all, Y_all, file_path)
                if processed_count > 0:
                    print(f"Processed {processed_count} samples from {file_path}")
            else:
                print(f"Unexpected data shape: {data.shape}")
                continue
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    if not X_all:
        raise ValueError("No valid data loaded from any file")
    X = np.concatenate(X_all, axis=0)
    F = np.concatenate(F_all, axis=0)
    Y = np.concatenate(Y_all, axis=0)
    # Truncate to 10,000 samples
    if X.shape[0] < 10000:
        raise ValueError(f"Insufficient samples: only {X.shape[0]} available, need 10000")
    X = X[:10000]
    F = F[:10000]
    Y = Y[:10000]
    print(f"Truncated data to 10000 samples: X={X.shape}, F={F.shape}, Y={Y.shape}")
    return X, F, Y

# 2. Hybrid Transformer-CNN Model
class HybridTransformerCNN(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_transformer_layers, num_cnn_filters=128, dropout=0.3):
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
            nn.Linear(d_model + 7, d_model),  # 7 features
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )
        self.residual = nn.Linear(d_model + 7, 2)

    def forward(self, x, features):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_projection(x) * np.sqrt(self.d_model)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = torch.cat((x, features), dim=1)
        out = self.fc(x) + self.residual(x)
        return out

# 3. Loss Function
def hybrid_loss(y_pred, y_true):
    mae = torch.mean(torch.abs(y_pred - y_true))
    mse = torch.mean((y_pred - y_true) ** 2)
    return 0.7 * mae + 0.3 * mse

# 4. Training Loop
def train_model(model, train_loader, val_loader, device, num_epochs=100, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs('bp_model_checkpoints', exist_ok=True)
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        for X_batch, F_batch, y_batch in train_loader:
            X_batch, F_batch, y_batch = X_batch.to(device), F_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch, F_batch)
            loss = hybrid_loss(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0
        val_mae_sbp = 0
        val_mae_dbp = 0
        with torch.no_grad():
            for X_batch, F_batch, y_batch in val_loader:
                X_batch, F_batch, y_batch = X_batch.to(device), F_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch, F_batch)
                loss = hybrid_loss(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_mae_sbp += torch.abs(y_pred[:, 0] - y_batch[:, 0]).sum().item()
                val_mae_dbp += torch.abs(y_pred[:, 1] - y_batch[:, 1]).sum().item()
        val_loss /= len(val_loader.dataset)
        val_mae_sbp /= len(val_loader.dataset)
        val_mae_dbp /= len(val_loader.dataset)
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}, '
              f'Val SBP MAE: {val_mae_sbp:.2f} mmHg, Val DBP MAE: {val_mae_dbp:.2f} mmHg, '
              f'Time: {epoch_time:.2f}s, Memory: {get_memory_usage():.1f} MB')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'bp_model_checkpoints/best_bp_hybrid_model.pth')
            print(f'Model saved at epoch {epoch+1} with validation loss: {val_loss:.2f}')
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
            pred = model(X_batch, F_batch)
            y_true.extend(y_batch.numpy())
            y_pred.extend(pred.cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae_sbp = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))
    mae_dbp = np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]))
    rmse_sbp = np.sqrt(np.mean((y_true[:, 0] - y_pred[:, 0])**2))
    rmse_dbp = np.sqrt(np.mean((y_true[:, 1] - y_pred[:, 1])**2))
    avg_actual_sbp = np.mean(y_true[:, 0])
    avg_predicted_sbp = np.mean(y_pred[:, 0])
    avg_actual_dbp = np.mean(y_true[:, 1])
    avg_predicted_dbp = np.mean(y_pred[:, 1])
    print(f'\n=== Test Results ===')
    print(f'Test SBP MAE: {mae_sbp:.2f} mmHg, RMSE: {rmse_sbp:.2f} mmHg')
    print(f'Test DBP MAE: {mae_dbp:.2f} mmHg, RMSE: {rmse_dbp:.2f} mmHg')
    print(f'Average Actual SBP: {avg_actual_sbp:.2f} mmHg')
    print(f'Average Predicted SBP: {avg_predicted_sbp:.2f} mmHg')
    print(f'Average Actual DBP: {avg_actual_dbp:.2f} mmHg')
    print(f'Average Predicted DBP: {avg_predicted_dbp:.2f} mmHg')
    print(f'=== End of Test Results ===\n')
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 2, 1)
    plt.plot(y_true[:100, 0], label='Actual SBP (mmHg)', color='green')
    plt.plot(y_pred[:100, 0], label='Predicted SBP (mmHg)', color='red', linestyle='--')
    plt.title('Actual vs Predicted SBP (First 100 Sequences)')
    plt.xlabel('Sequence Index')
    plt.ylabel('SBP (mmHg)')
    plt.legend()
    plt.subplot(3, 2, 2)
    plt.plot(y_true[:100, 1], label='Actual DBP (mmHg)', color='green')
    plt.plot(y_pred[:100, 1], label='Predicted DBP (mmHg)', color='red', linestyle='--')
    plt.title('Actual vs Predicted DBP (First 100 Sequences)')
    plt.xlabel('Sequence Index')
    plt.ylabel('DBP (mmHg)')
    plt.legend()
    plt.subplot(3, 2, 3)
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_true[:, 0].min(), y_true[:, 0].max()], [y_true[:, 0].min(), y_true[:, 0].max()], 'r--', lw=2)
    plt.title(f'SBP: Predicted vs Actual (MAE={mae_sbp:.2f} mmHg)')
    plt.xlabel('Actual SBP (mmHg)')
    plt.ylabel('Predicted SBP (mmHg)')
    plt.subplot(3, 2, 4)
    plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_true[:, 1].min(), y_true[:, 1].max()], [y_true[:, 1].max(), y_true[:, 1].min()], 'r--', lw=2)
    plt.title(f'DBP: Predicted vs Actual (MAE={mae_dbp:.2f} mmHg)')
    plt.xlabel('Actual DBP (mmHg)')
    plt.ylabel('Predicted DBP (mmHg)')
    plt.subplot(3, 2, 5)
    errors_sbp = y_pred[:, 0] - y_true[:, 0]
    plt.hist(errors_sbp, bins=50, color='blue', alpha=0.7)
    plt.title('SBP Error Distribution')
    plt.xlabel('Error (mmHg)')
    plt.ylabel('Frequency')
    plt.subplot(3, 2, 6)
    errors_dbp = y_pred[:, 1] - y_true[:, 1]
    plt.hist(errors_dbp, bins=50, color='blue', alpha=0.7)
    plt.title('DBP Error Distribution')
    plt.xlabel('Error (mmHg)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    return mae_sbp, mae_dbp, avg_actual_sbp, avg_predicted_sbp, avg_actual_dbp, avg_predicted_dbp

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
    data_files = ['part_1.mat']
    window_size = 250
    X, F, Y = load_and_preprocess_data(data_files, window_size=window_size, fs=125)
    print(f'Memory after data loading: {get_memory_usage():.1f} MB')
    print(f'Data shapes: X={X.shape}, F={F.shape}, Y={Y.shape}')
    if X.shape[0] == 0 or Y.shape[0] == 0:
        print("Error: No valid data extracted!")
        return
    print(f'X range: [{X.min():.3f}, {X.max():.3f}]')
    print(f'Y SBP range: [{Y[:, 0].min():.3f}, {Y[:, 0].max():.3f}]')
    print(f'Y DBP range: [{Y[:, 1].min():.3f}, {Y[:, 1].max():.3f}]')
    print(f'Y SBP mean: {Y[:, 0].mean():.3f}, Y SBP std: {Y[:, 0].std():.3f}')
    print(f'Y DBP mean: {Y[:, 1].mean():.3f}, Y DBP std: {Y[:, 1].std():.3f}')
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
    batch_size = 16
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
        d_model=256,
        nhead=4,
        num_transformer_layers=4,
        num_cnn_filters=128,
        dropout=0.3
    ).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Memory after model creation: {get_memory_usage():.1f} MB')
    print('Starting training...')
    start_time = time.time()
    train_model(model, train_loader, val_loader, device, num_epochs=100, patience=10)
    print(f'Training completed in {time.time() - start_time:.2f} seconds')
    print('Loading best model from bp_model_checkpoints/best_bp_hybrid_model.pth')
    model.load_state_dict(torch.load('bp_model_checkpoints/best_bp_hybrid_model.pth'))
    print('\nEvaluating on test set...')
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()