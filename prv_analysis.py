import wfdb
import numpy as np
from scipy import signal
import pywt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import os
import time
import psutil
import gc
import warnings
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')

# 1. Advanced Preprocessing
def denoise_signal(signal_data, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs, wavelet)
    return signal.medfilt(denoised, kernel_size=5)

def signal_quality_check(signal_data, fs=125, threshold=0.1):
    if np.std(signal_data) < threshold or np.any(~np.isfinite(signal_data)):
        return False
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=min(len(signal_data), 500))
    high_freq_power = np.sum(psd[freqs > 5.0]) / np.sum(psd)
    return high_freq_power <= 0.5

def detect_peaks(signal_data, fs, min_distance=0.4, signal_type='ppg'):
    if not signal_quality_check(signal_data, fs):
        return np.array([])
    
    signal_data = denoise_signal(signal_data)
    
    # Adaptive prominence based on signal amplitude
    signal_std = np.std(signal_data)
    if signal_type == 'ecg':
        prominences = [signal_std * f for f in [1.0, 0.5, 0.3, 0.1]]
        for prom in prominences:
            peaks, _ = signal.find_peaks(signal_data, distance=int(fs * min_distance), prominence=prom)
            if len(peaks) >= 3:
                return peaks
            peaks, _ = signal.find_peaks(-signal_data, distance=int(fs * min_distance), prominence=prom)
            if len(peaks) >= 3:
                return peaks
    else:
        prominences = [signal_std * f for f in [0.5, 0.3, 0.2]]
        for prom in prominences:
            peaks, _ = signal.find_peaks(signal_data, distance=int(fs * min_distance), prominence=prom)
            if len(peaks) >= 3:
                return peaks
    return np.array([])

# 2. PRV/HRV Feature Extraction
def extract_prv_hrv_features(signal_data, peaks, fs, signal_type='ppg'):
    if len(peaks) < 3:
        return np.zeros(6)
    
    ibis = np.diff(peaks) / fs * 1000  # Convert to milliseconds
    if len(ibis) < 2:
        return np.zeros(6)
    
    # Robust outlier removal
    median_ibi = np.median(ibis)
    mad = np.median(np.abs(ibis - median_ibi)) / 0.6745  # Normalized MAD
    ibis = ibis[np.abs(ibis - median_ibi) <= 3 * mad]
    ibis = ibis[(ibis >= 200) & (ibis <= 2000)]  # 30-300 bpm
    
    if len(ibis) < 2:
        return np.zeros(6)
    
    # Time-domain features
    rmssd = np.sqrt(np.mean(np.diff(ibis) ** 2))
    sdnn = np.std(ibis)
    pnn50 = 100 * np.sum(np.abs(np.diff(ibis)) > 50) / len(ibis)
    
    # Frequency-domain features with power normalization
    # Create proper time array: start at 0, then cumulative sum of IBIs
    t = np.zeros(len(ibis) + 1)
    t[1:] = np.cumsum(ibis) / 1000  # Time in seconds
    
    # Create uniform time grid
    t_uniform = np.arange(t[0], t[-1], 0.25)  # 4 Hz
    if len(t) < 3 or len(t_uniform) < 8:  # Ensure enough samples
        return np.zeros(6)
    
    # Use midpoint times for each IBI interval - this ensures proper length matching
    t_mid = (t[:-1] + t[1:]) / 2  # This has length len(ibis)
    
    # Ensure arrays match in length before interpolation
    if len(t_mid) != len(ibis):
        return np.zeros(6)
    
    try:
        interp_func = interp1d(t_mid, ibis, kind='linear', fill_value='extrapolate', bounds_error=False)
        ibis_uniform = interp_func(t_uniform)
        
        # Check for valid interpolation result
        if np.any(np.isnan(ibis_uniform)) or len(ibis_uniform) < 8:
            return np.zeros(6)
    except Exception:
        return np.zeros(6)
    
    freqs, psd = signal.welch(ibis_uniform, fs=4, nperseg=min(len(ibis_uniform), 64), scaling='density')
    total_power = np.trapz(psd, freqs)
    lf_band = (freqs >= 0.04) & (freqs < 0.15)
    hf_band = (freqs >= 0.15) & (freqs < 0.4)
    lf_power = np.trapz(psd[lf_band], freqs[lf_band]) / (total_power + 1e-8) if np.any(lf_band) else 0
    hf_power = np.trapz(psd[hf_band], freqs[hf_band]) / (total_power + 1e-8) if np.any(hf_band) else 0
    
    # Nonlinear feature: Approximate Entropy
    def ap_entropy(U, m=2, r=0.2):
        try:
            def _phi(m):
                x = [U[i:i+m] for i in range(len(U)-m+1)]
                C = [np.sum(np.max(np.abs(u - x), axis=1) <= r) / (len(U)-m+1) for u in x]
                C = np.array(C)
                C = C[C > 0]
                return np.sum(np.log(C)) / len(C) if len(C) > 0 else 0
            m = min(m, len(U)-1)
            phi_m = _phi(m)
            phi_m1 = _phi(m+1)
            ap_en = abs(phi_m1 - phi_m)
            return 0.0 if np.isnan(ap_en) or np.isinf(ap_en) else ap_en
        except:
            return 0.0
    
    ap_en = ap_entropy(ibis, m=2, r=0.2 * np.std(ibis) if np.std(ibis) > 0 else 0.1)
    
    return np.array([rmssd, sdnn, pnn50, lf_power, hf_power, ap_en])

def extract_features(ecg, ppg, fs):
    ecg_peaks = detect_peaks(ecg, fs, signal_type='ecg')
    ppg_peaks = detect_peaks(ppg, fs, signal_type='ppg')
    prv_features = extract_prv_hrv_features(ppg, ppg_peaks, fs, 'ppg')
    hrv_features = extract_prv_hrv_features(ecg, ecg_peaks, fs, 'ecg')
    
    if np.random.rand() < 0.01:
        print(f"ECG peaks: {len(ecg_peaks)}, PPG peaks: {len(ppg_peaks)}")
        if len(ecg_peaks) > 1:
            ibis_ecg = np.diff(ecg_peaks) / fs * 1000
            print(f"ECG IBIs: min={ibis_ecg.min():.1f}, max={ibis_ecg.max():.1f}, mean={ibis_ecg.mean():.1f}")
        if len(ppg_peaks) > 1:
            ibis_ppg = np.diff(ppg_peaks) / fs * 1000
            print(f"PPG IBIs: min={ibis_ppg.min():.1f}, max={ibis_ppg.max():.1f}, mean={ibis_ppg.mean():.1f}")
    
    return prv_features, hrv_features

# 3. Load and Preprocess BIDMC Dataset
def load_and_preprocess_data(window_size=1000):
    records = ['bidmc01', 'bidmc02', 'bidmc03', 'bidmc04', 'bidmc05']
    X_all, F_all, Y_all = [], [], []
    feature_scaler = RobustScaler()
    
    for record_name in records:
        try:
            print(f"Loading record: {record_name}")
            record = wfdb.rdrecord(record_name, pn_dir='bidmc/1.0.0/')
            fs = record.fs
            ecg = record.p_signal[:, 0]
            ppg = record.p_signal[:, 1]
            
            ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
            ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
            
            step_size = window_size // 2
            num_samples = (len(ecg) - window_size) // step_size + 1
            X = np.zeros((num_samples, window_size, 2))
            F = np.zeros((num_samples, 6))
            Y = np.zeros((num_samples, 6))
            
            for i in range(num_samples):
                start = i * step_size
                end = start + window_size
                X[i, :, 0] = ecg[start:end]
                X[i, :, 1] = ppg[start:end]
                prv_features, hrv_features = extract_features(ecg[start:end], ppg[start:end], fs)
                F[i] = prv_features
                Y[i] = hrv_features
            
            valid_idx = (Y[:, 0] > 0) & (Y[:, 1] > 0) & np.all(np.isfinite(Y), axis=1)
            if np.sum(valid_idx) > 0:
                # Clip extreme values before scaling
                F[valid_idx] = np.clip(F[valid_idx], np.percentile(F[valid_idx], 1, axis=0), 
                                       np.percentile(F[valid_idx], 99, axis=0))
                Y[valid_idx] = np.clip(Y[valid_idx], np.percentile(Y[valid_idx], 1, axis=0), 
                                       np.percentile(Y[valid_idx], 99, axis=0))
                F[valid_idx] = feature_scaler.fit_transform(F[valid_idx])
                Y[valid_idx] = feature_scaler.transform(Y[valid_idx])
                X_all.append(X[valid_idx])
                F_all.append(F[valid_idx])
                Y_all.append(Y[valid_idx])
                print(f"Added {np.sum(valid_idx)}/{num_samples} valid windows from {record_name}")
        except Exception as e:
            print(f"Error loading {record_name}: {e}")
            continue
    
    if not X_all:
        raise Exception("No valid data loaded from any record")
    
    X = np.concatenate(X_all, axis=0)
    F = np.concatenate(F_all, axis=0)
    Y = np.concatenate(Y_all, axis=0)
    return X, F, Y, feature_scaler

# 4. Hybrid Transformer-CNN Model
class HybridTransformerCNN(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_transformer_layers, num_cnn_filters=64, dropout=0.3):
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
            nn.Linear(d_model + 6, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 6)
        )
        self.residual = nn.Linear(d_model + 6, 6)

    def forward(self, x, features):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_projection(x) * np.sqrt(self.d_model)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = torch.cat((x, features), dim=1)
        out = self.fc(x) + self.residual(x)
        return out

# 5. Custom Loss Function
def custom_mae_loss(y_pred, y_true):
    mae = torch.abs(y_pred - y_true)
    # Weight features inversely by their standard deviation
    weights = torch.tensor([1.0, 1.0, 0.5, 2.0, 2.0, 1.0], device=y_true.device)
    return torch.mean(mae * weights)

# 6. Training Loop
def train_model(model, train_loader, val_loader, device, num_epochs=50, patience=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    os.makedirs('prv_model_checkpoints', exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, F_batch, y_batch in train_loader:
            X_batch, F_batch, y_batch = X_batch.to(device), F_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch, F_batch)
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
                y_pred = model(X_batch, F_batch)
                loss = custom_mae_loss(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_mae += torch.abs(y_pred - y_batch).sum().item()
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset) * 6
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.2f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'prv_model_checkpoints/best_prv_hybrid_model.pth')
            print(f'Model saved at epoch {epoch+1} with validation loss: {val_loss:.6f}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        scheduler.step()
        
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    return best_val_loss

# 7. Evaluation and Visualization
def evaluate_model(model, test_loader, device, feature_scaler):
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
    
    y_true_orig = feature_scaler.inverse_transform(y_true)
    y_pred_orig = feature_scaler.inverse_transform(y_pred)
    
    mae = np.mean(np.abs(y_true_orig - y_pred_orig), axis=0)
    feature_names = ['RMSSD', 'SDNN', 'pNN50', 'LF Power', 'HF Power', 'ApEn']
    
    print(f'\n=== Test Results ===')
    print(f"{'Feature':<15} {'MAE':>10} {'Actual Mean':>12} {'Predicted Mean':>15} {'Correlation':>12}")
    print('-' * 65)
    correlations = []
    for i, name in enumerate(feature_names):
        actual_mean = np.mean(y_true_orig[:, i])
        pred_mean = np.mean(y_pred_orig[:, i])
        corr, _ = pearsonr(y_true_orig[:, i], y_pred_orig[:, i])
        correlations.append(corr)
        print(f'{name:<15} {mae[i]:>10.2f} {actual_mean:>12.2f} {pred_mean:>15.2f} {corr:>12.2f}')
    print(f'\nAverage MAE: {np.mean(mae):.2f}')
    print(f'Average Correlation: {np.mean(correlations):.2f}')
    print(f'=== End of Test Results ===\n')
    
    # Plotting
    plt.figure(figsize=(15, 12))
    for i, name in enumerate(feature_names):
        plt.subplot(3, 2, i+1)
        plt.scatter(y_true_orig[:, i], y_pred_orig[:, i], alpha=0.5)
        plt.plot([y_true_orig[:, i].min(), y_true_orig[:, i].max()], 
                 [y_true_orig[:, i].min(), y_true_orig[:, i].max()], 'r--', lw=2)
        plt.title(f'{name} (MAE={mae[i]:.2f}, Corr={correlations[i]:.2f})')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
    
    plt.tight_layout()
    plt.show()
    
    # Time series plot
    plt.figure(figsize=(15, 12))
    for i, name in enumerate(feature_names):
        plt.subplot(3, 2, i+1)
        plt.plot(y_true_orig[:100, i], label='Actual', color='blue')
        plt.plot(y_pred_orig[:100, i], label='Predicted', color='red', linestyle='--')
        plt.title(f'{name} (First 100 Samples)')
        plt.xlabel('Sample Index')
        plt.ylabel(name)
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Error histograms
    plt.figure(figsize=(15, 12))
    for i, name in enumerate(feature_names):
        plt.subplot(3, 2, i+1)
        errors = y_pred_orig[:, i] - y_true_orig[:, i]
        plt.hist(errors, bins=50, alpha=0.7, color='purple')
        plt.title(f'{name} Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return mae, y_true_orig, y_pred_orig

# 8. Main Function
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    print(f'Initial memory usage: {get_memory_usage():.1f} MB')
    
    # Load and preprocess data
    window_size = 1000
    X, F, Y, feature_scaler = load_and_preprocess_data(window_size=window_size)
    
    print(f'Memory after data loading: {get_memory_usage():.1f} MB')
    cleanup_memory()
    
    # Data stats
    print(f'Data shapes: X={X.shape}, F={F.shape}, Y={Y.shape}')
    if X.shape[0] == 0 or Y.shape[0] == 0:
        print("Error: No valid data extracted!")
        return
    print(f'X range: [{X.min():.3f}, {X.max():.3f}]')
    feature_names = ['RMSSD', 'SDNN', 'pNN50', 'LF Power', 'HF Power', 'ApEn']
    Y_orig = feature_scaler.inverse_transform(Y)
    for i, name in enumerate(feature_names):
        print(f'{name} range: [{Y_orig[:, i].min():.3f}, {Y_orig[:, i].max():.3f}], '
              f'mean: {Y_orig[:, i].mean():.3f}, std: {Y_orig[:, i].std():.3f}')
    
    # Split dataset
    X_temp, X_test, F_temp, F_test, Y_temp, Y_test = train_test_split(X, F, Y, test_size=0.15, random_state=42)
    X_train, X_val, F_train, F_val, Y_train, Y_val = train_test_split(X_temp, F_temp, Y_temp, test_size=0.15/0.85, random_state=42)
    print(f'Dataset sizes: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}')
    
    # Convert to PyTorch tensors
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
    
    # Create DataLoaders
    batch_size = 16
    train_dataset = TensorDataset(X_train, F_train, Y_train)
    val_dataset = TensorDataset(X_val, F_val, Y_val)
    test_dataset = TensorDataset(X_test, F_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f'Using batch size: {batch_size}')
    print(f'Memory after DataLoader creation: {get_memory_usage():.1f} MB')
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridTransformerCNN(
        input_size=2, 
        d_model=128, 
        nhead=4, 
        num_transformer_layers=3, 
        num_cnn_filters=64, 
        dropout=0.3
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Memory after model creation: {get_memory_usage():.1f} MB')
    
    # Train model
    print('Starting training...')
    start_time = time.time()
    train_model(model, train_loader, val_loader, device, num_epochs=50, patience=15)
    print(f'Training completed in {time.time() - start_time:.2f} seconds')
    
    # Load best model
    print('Loading best model from prv_model_checkpoints/best_prv_hybrid_model.pth')
    model.load_state_dict(torch.load('prv_model_checkpoints/best_prv_hybrid_model.pth'))
    
    # Evaluate on test set
    print('\nEvaluating on test set...')
    mae, y_true_orig, y_pred_orig = evaluate_model(model, test_loader, device, feature_scaler)

if __name__ == "__main__":
    main()