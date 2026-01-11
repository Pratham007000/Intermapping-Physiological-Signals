import scipy.io
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
        return signal.medfilt(denoised[:len(signal_data)], kernel_size=5)
    except:
        # Fallback to simple median filtering if wavelet denoising fails
        return signal.medfilt(signal_data, kernel_size=5)

def signal_quality_check(signal_data, threshold=0.01, debug=False):
    signal_std = np.std(signal_data)
    has_invalid = np.any(~np.isfinite(signal_data))
    
    if debug:
        print(f"Signal std: {signal_std:.4f}, threshold: {threshold}, has_invalid: {has_invalid}")
    
    if signal_std < threshold or has_invalid:
        if debug:
            print(f"Failed std/invalid check: std={signal_std:.4f} < {threshold} or has_invalid={has_invalid}")
        return False
    
    freqs, psd = signal.welch(signal_data, fs=128, nperseg=min(len(signal_data), 512))
    high_freq_power = np.sum(psd[freqs > 1.0]) / np.sum(psd)
    
    if debug:
        print(f"High freq power ratio: {high_freq_power:.4f}")
    
    if high_freq_power > 0.7:
        if debug:
            print(f"Failed high freq check: {high_freq_power:.4f} > 0.7")
        return False
    return True

# Extract Enhanced Features from ECG
def extract_features(ecg, fs):
    ecg_peaks, _ = signal.find_peaks(ecg, distance=int(fs*0.4), prominence=0.5)
    rsa_feature = np.std(np.diff(ecg_peaks) / fs) if len(ecg_peaks) >= 2 else 0
    
    freqs, psd = signal.welch(ecg, fs=fs, nperseg=min(len(ecg), fs*4))
    lf_band = (freqs >= 0.04) & (freqs <= 0.15)
    hf_band = (freqs >= 0.15) & (freqs <= 0.4)
    lf_power = np.sum(psd[lf_band]) if np.any(lf_band) else 0
    hf_power = np.sum(psd[hf_band]) if np.any(hf_band) else 0
    lfhf_ratio = lf_power / (hf_power + 1e-8)
    
    # Additional HRV features
    rr_intervals = np.diff(ecg_peaks) / fs if len(ecg_peaks) >= 2 else np.array([0])
    sdnn = np.std(rr_intervals) if len(rr_intervals) > 0 else 0
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) if len(rr_intervals) > 1 else 0
    
    try:
        scales = np.arange(1, 32)
        coeffs, _ = pywt.cwt(ecg[:min(len(ecg), 1000)], scales, 'morl', sampling_period=1/fs)
        cwt_feature = np.std(coeffs) if coeffs.size > 0 else 0
    except:
        cwt_feature = 0
    
    amp_feature = np.std(ecg[ecg_peaks]) if len(ecg_peaks) >= 2 else 0
    
    return np.array([rsa_feature, lfhf_ratio, cwt_feature, amp_feature, sdnn, rmssd])

# Load and preprocess dataset for multiple specific files
def load_and_preprocess_data(window_size=500, base_dir='~/Desktop/PPG_Estimation_Project/g2p7vwxyn2-1/ECG_GSR_Emotions/Raw Data/Multimodal'):
    base_dir = os.path.expanduser(base_dir)
    X, F, Y = [], [], []
    fs = 128
    
    ecg_dir = os.path.join(base_dir, 'ECG')
    gsr_dir = os.path.join(base_dir, 'GSR')
    
    # List of target IDs to process
    target_ids = ['s1p1v1', 's1p1v2']
    
    for target_id in target_ids:
        ecg_file = os.path.join(ecg_dir, f'ECGdata_{target_id}.mat')
        gsr_file = os.path.join(gsr_dir, f'GSRdata_{target_id}.mat')
        
        if not os.path.exists(ecg_file) or not os.path.exists(gsr_file):
            print(f"Warning: Required files not found for {target_id}: ECGdata_{target_id}.mat or GSRdata_{target_id}.mat. Skipping.")
            continue
        
        print(f"\nLoading matching data for subject: {target_id}")
        print(f"ECG file: ECGdata_{target_id}.mat")
        print(f"GSR file: GSRdata_{target_id}.mat")
        
        try:
            ecg_data = scipy.io.loadmat(ecg_file)
            gsr_data = scipy.io.loadmat(gsr_file)
            
            ecg_keys = [k for k in ecg_data.keys() if not k.startswith('__')]
            gsr_keys = [k for k in gsr_data.keys() if not k.startswith('__')]
            
            print(f"Available ECG keys: {ecg_keys}")
            print(f"Available GSR keys: {gsr_keys}")
            
            ecg = None
            for key in ['ECGdata', 'ecg', 'ECG', 'data']:
                if key in ecg_data:
                    ecg = ecg_data[key]
                    break
            
            gsr = None
            for key in ['GSRdata', 'gsr', 'GSR', 'data']:
                if key in gsr_data:
                    gsr = gsr_data[key]
                    break
            
            if ecg is None:
                ecg = ecg_data[ecg_keys[0]] if ecg_keys else []
            if gsr is None:
                gsr = gsr_data[gsr_keys[0]] if gsr_keys else []
            
            if isinstance(ecg, np.ndarray) and ecg.ndim > 1:
                ecg = ecg.flatten()
            if isinstance(gsr, np.ndarray) and gsr.ndim > 1:
                gsr = gsr.flatten()
            
            ecg = np.array(ecg).flatten()
            gsr = np.array(gsr).flatten()
            
        except Exception as e:
            print(f"Error loading data for subject {target_id}: {e}")
            continue
        
        if len(ecg) == 0 or len(gsr) == 0:
            print(f"No valid ECG or GSR data found for subject {target_id}. Skipping.")
            continue
        
        min_length = min(len(ecg), len(gsr))
        ecg = ecg[:min_length]
        gsr = gsr[:min_length]
        
        # Advanced preprocessing with denoising
        ecg = denoise_signal(ecg)
        gsr = denoise_signal(gsr)
        
        # Robust normalization using quantiles
        ecg_q25, ecg_q75 = np.percentile(ecg, [25, 75])
        gsr_q25, gsr_q75 = np.percentile(gsr, [25, 75])
        ecg_iqr = ecg_q75 - ecg_q25 + 1e-8
        gsr_iqr = gsr_q75 - gsr_q25 + 1e-8
        
        ecg = (ecg - np.median(ecg)) / ecg_iqr
        gsr = (gsr - np.median(gsr)) / gsr_iqr
        
        # Clip outliers
        ecg = np.clip(ecg, -3, 3)
        gsr = np.clip(gsr, -3, 3)
        
        step_size = window_size // 2
        num_samples = (len(ecg) - window_size) // step_size + 1
        X_subject = np.zeros((num_samples, window_size, 1))
        F_subject = np.zeros((num_samples, 6))
        Y_subject = np.zeros((num_samples, window_size))
        
        print(f"Processing {num_samples} windows of size {window_size} for {target_id}...")
        valid_windows = 0
        
        for i in range(num_samples):
            start = i * step_size
            end = start + window_size
            
            ecg_segment = ecg[start:end]
            gsr_segment = gsr[start:end]
            
            debug_mode = i < 5 and target_id == 's1p1v1'  # Debug only for first subject/file
            if debug_mode:
                print(f"\nWindow {i+1}/{num_samples} (samples {start}-{end}):")
            
            ecg_valid = signal_quality_check(ecg_segment, threshold=0.01, debug=debug_mode)
            gsr_valid = signal_quality_check(gsr_segment, threshold=0.01, debug=debug_mode)
            
            if debug_mode:
                print(f"ECG valid: {ecg_valid}, GSR valid: {gsr_valid}")
            
            if ecg_valid and gsr_valid:
                # Data augmentation: Add small noise
                noise = np.random.normal(0, 0.01, window_size)
                X_subject[i, :, 0] = ecg_segment + noise
                F_subject[i] = extract_features(ecg_segment, fs)
                Y_subject[i] = gsr_segment
                valid_windows += 1
                if debug_mode:
                    print(f"Window {i+1} accepted (total valid: {valid_windows})")
            elif debug_mode:
                print(f"Window {i+1} rejected")
        
        valid_idx = np.any(Y_subject != 0, axis=1)
        print(f"Total valid windows for {target_id}: {valid_windows}/{num_samples} ({100*valid_windows/num_samples:.1f}%)")
        
        if np.sum(valid_idx) > 0:
            X_subject = X_subject[valid_idx]
            F_subject = F_subject[valid_idx]
            Y_subject = Y_subject[valid_idx]
            X.append(X_subject)
            F.append(F_subject)
            Y.append(Y_subject)
        else:
            print(f"No valid data windows extracted for {target_id}. Skipping.")
    
    if not X:
        raise Exception("No valid data windows extracted from any file.")
    
    # Concatenate data from all files
    X = np.concatenate(X, axis=0)
    F = np.concatenate(F, axis=0)
    Y = np.concatenate(Y, axis=0)
    print(f"Final dataset shape across all files: X={X.shape}, F={F.shape}, Y={Y.shape}")
    
    return X, F, Y

# 2. Enhanced Hybrid Transformer-CNN Model
class EnhancedHybridModel(nn.Module):
    def __init__(self, input_size=1, d_model=256, nhead=8, num_transformer_layers=4, num_cnn_filters=128, dropout=0.6, window_size=500):
        super(EnhancedHybridModel, self).__init__()
        self.d_model = d_model
        self.window_size = window_size
        
        # Add LSTM layer
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)

        # Increase CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, num_cnn_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(num_cnn_filters),
            nn.Conv1d(num_cnn_filters, num_cnn_filters, kernel_size=5, padding=2),
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

        # Update FC layer dimensions to match LSTM output
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2 + 6, d_model),  # Updated to 6 features
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, window_size)
        )
        self.residual = nn.Linear(d_model * 2 + 6, window_size)
        

    def forward(self, x, features):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_projection(x) * np.sqrt(self.d_model)
        x = self.transformer(x)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        x = torch.cat((x, features), dim=1)
        out = self.fc(x) + self.residual(x)
        return out

# 3. Enhanced Custom Loss Function with Temporal Consistency
def custom_mae_loss(y_pred, y_true, lambda_baseline=0.1, lambda_temporal=0.05):
    mae = torch.abs(y_pred - y_true)
    weights = 1.0 + (mae / 2.0)
    baseline_loss = lambda_baseline * torch.abs(torch.mean(y_pred, dim=1) - torch.mean(y_true, dim=1))
    
    # Temporal consistency loss (smoothness)
    temporal_loss = lambda_temporal * torch.mean(torch.abs(y_pred[:, 1:] - y_pred[:, :-1]))
    
    return torch.mean(mae * weights) + baseline_loss.mean() + temporal_loss

# 4. Training Loop
def train_model(model, train_loader, val_loader, device, num_epochs=100, patience=30):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, total_steps=num_epochs)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    os.makedirs('gsr_model_checkpoints', exist_ok=True)
    
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
                val_mae += torch.abs(y_pred - y_batch).sum().item() / y_batch.size(1)
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'gsr_model_checkpoints/best_gsr_hybrid_model.pth')
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
            pred = model(X_batch, F_batch)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    
    print(f'\n=== Test Results ===')
    print(f'Test MAE: {mae:.4f}')
    print(f'Average Actual GSR: {np.mean(y_true):.4f}')
    print(f'Average Predicted GSR: {np.mean(y_pred):.4f}')
    print(f'=== End of Test Results ===\n')
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(y_true[0], label='Actual GSR', color='green')
    plt.plot(y_pred[0], label='Predicted GSR', color='red', linestyle='--')
    plt.title('Actual vs Predicted GSR (First Sequence)')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized GSR')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f'Predicted vs Actual GSR (MAE={mae:.4f})')
    plt.xlabel('Actual GSR')
    plt.ylabel('Predicted GSR')
    
    plt.subplot(3, 1, 3)
    errors = y_pred - y_true
    plt.hist(errors.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Error Distribution (Predicted - Actual GSR)')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return mae

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
    base_dir = '~/Desktop/PPG_Estimation_Project/g2p7vwxyn2-1/ECG_GSR_Emotions/Raw Data/Multimodal'
    X, F, Y = load_and_preprocess_data(window_size=window_size, base_dir=base_dir)
    
    print(f'Memory after data loading: {get_memory_usage():.1f} MB')
    cleanup_memory()
    
    print(f'Data shapes: X={X.shape}, F={F.shape}, Y={Y.shape}')
    if X.shape[0] == 0 or Y.shape[0] == 0:
        print("Error: No valid data extracted!")
        return
    print(f'X range: [{X.min():.3f}, {X.max():.3f}]')
    print(f'Y range: [{Y.min():.3f}, {Y.max():.3f}]')
    print(f'Y mean: {Y.mean():.3f}, Y std: {Y.std():.3f}')
    
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
    model = EnhancedHybridModel(
        input_size=1, 
        d_model=256,
        nhead=8,
        num_transformer_layers=4,
        num_cnn_filters=128,
        dropout=0.6,
        window_size=window_size
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Memory after model creation: {get_memory_usage():.1f} MB')
    
    print('Starting training...')
    start_time = time.time()
    train_model(model, train_loader, val_loader, device, num_epochs=100, patience=30)
    print(f'Training completed in {time.time() - start_time:.2f} seconds')
    
    print('Loading best model from gsr_model_checkpoints/best_gsr_hybrid_model.pth')
    model.load_state_dict(torch.load('gsr_model_checkpoints/best_gsr_hybrid_model.pth'))
    
    print('\nEvaluating on test set...')
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()