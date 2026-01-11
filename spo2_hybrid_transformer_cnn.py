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
import logging
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        logging.warning("Signal rejected: low std or non-finite values")
        return False
    freqs, psd = signal.welch(signal_data, fs=125, nperseg=min(len(signal_data), 500))
    high_freq_power = np.sum(psd[freqs > 1.0]) / np.sum(psd)
    if high_freq_power > 0.5:
        logging.warning("Signal rejected: excessive high-frequency noise")
        return False
    return True

def extract_spo2(ppg, fs, spo2_annotations=None):
    if len(ppg) < fs or not signal_quality_check(ppg):
        return 0
    
    if spo2_annotations is not None:
        # Use actual SpO2 annotations if available (replace with your data source)
        spo2 = np.mean(spo2_annotations)  # Example: average SpO2 for the window
        return max(90, min(100, spo2))
    
    # Fallback synthetic SpO2 calculation (for demonstration)
    ppg = denoise_signal(ppg)
    sos = signal.butter(4, [0.5, 5.0], btype='band', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, ppg)
    filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    
    peaks, _ = signal.find_peaks(filtered, distance=int(fs*0.4), prominence=0.5)
    troughs, _ = signal.find_peaks(-filtered, distance=int(fs*0.4), prominence=0.5)
    
    if len(peaks) < 2 or len(troughs) < 2:
        logging.warning("Not enough peaks/troughs for SpO2 calculation")
        return 0
    
    ac_component = np.mean(filtered[peaks]) - np.mean(filtered[troughs])
    dc_component = np.mean(filtered)
    if dc_component == 0:
        logging.warning("Zero DC component in PPG")
        return 0
    
    # Placeholder SpO2 calculation (replace with actual formula or data)
    ratio = ac_component / (dc_component + 1e-8)
    spo2 = 100 - 5 * ratio  # Adjusted for better range
    spo2 = max(90, min(100, spo2))
    return spo2

def extract_features(ecg, ppg, fs):
    ecg_peaks, _ = signal.find_peaks(ecg, distance=int(fs*0.4), prominence=0.5)
    rsa_feature = np.std(np.diff(ecg_peaks) / fs) if len(ecg_peaks) >= 2 else 0
    
    ppg_peaks, _ = signal.find_peaks(ppg, distance=int(fs*0.4), prominence=0.5)
    ppg_troughs, _ = signal.find_peaks(-ppg, distance=int(fs*0.4), prominence=0.5)
    ac_component = np.mean(ppg[ppg_peaks]) - np.mean(ppg[ppg_troughs]) if len(ppg_peaks) >= 2 and len(ppg_troughs) >= 2 else 0
    dc_component = np.mean(ppg)
    ac_dc_ratio = ac_component / (dc_component + 1e-8) if dc_component != 0 else 0
    
    freqs, psd = signal.welch(ppg, fs=fs, nperseg=min(len(ppg), fs*4))
    spo2_band = (freqs >= 0.5) & (freqs <= 5.0)
    freq_feature = np.sum(psd[spo2_band]) / (np.sum(psd) + 1e-8) if np.any(spo2_band) else 0
    
    try:
        scales = np.arange(1, 32)
        coeffs, _ = pywt.cwt(ppg[:min(len(ppg), 1000)], scales, 'morl', sampling_period=1/fs)
        if np.any(spo2_band):
            band_indices = np.where(spo2_band)[0]
            if len(band_indices) > 0:
                start_idx = max(0, band_indices[0])
                end_idx = min(coeffs.shape[1], band_indices[-1] + 1)
                cwt_feature = np.std(coeffs[:, start_idx:end_idx]) if start_idx < end_idx else 0
            else:
                cwt_feature = 0
        else:
            cwt_feature = 0
    except Exception as e:
        logging.error(f"CWT error: {e}")
        cwt_feature = 0
    
    return np.array([rsa_feature, ac_dc_ratio, freq_feature, cwt_feature])

def load_and_preprocess_data(window_size=500):
    records = ['bidmc01', 'bidmc02', 'bidmc03', 'bidmc04', 'bidmc05']
    X_all, F_all, Y_all = [], [], []
    
    for record_name in records:
        try:
            logging.info(f"Loading record: {record_name}")
            record = wfdb.rdrecord(record_name, pn_dir='bidmc/1.0.0/')
            fs = record.fs
            ecg = record.p_signal[:, 0]
            ppg = record.p_signal[:, 1]
            # Replace with actual SpO2 annotations if available
            spo2_annotations = None  # record.p_signal[:, 3] if available
            
            ecg = (ecg - np.mean(ecg)) / np.std(ecg)
            ppg = (ppg - np.mean(ppg)) / np.std(ppg)
            
            step_size = window_size // 2
            num_samples = (len(ecg) - window_size) // step_size + 1
            X = np.zeros((num_samples, window_size, 2))
            F = np.zeros((num_samples, 4))
            Y = np.zeros(num_samples)
            
            for i in range(num_samples):
                start = i * step_size
                end = start + window_size
                X[i, :, 0] = ecg[start:end]
                X[i, :, 1] = ppg[start:end]
                F[i] = extract_features(ecg[start:end], ppg[start:end], fs)
                Y[i] = extract_spo2(ppg[start:end], fs, spo2_annotations)
            
            valid_idx = (Y >= 90) & (Y <= 100)
            logging.info(f"Record {record_name}: {np.sum(valid_idx)} valid windows out of {num_samples}")
            if np.sum(valid_idx) > 0:
                X_all.append(X[valid_idx])
                F_all.append(F[valid_idx])
                Y_all.append(Y[valid_idx])
        except Exception as e:
            logging.error(f"Error loading {record_name}: {e}")
            continue
    
    if not X_all:
        raise Exception("No valid data loaded from any record")
    
    X = np.concatenate(X_all, axis=0)
    F = np.concatenate(F_all, axis=0)
    Y = np.concatenate(Y_all, axis=0)
    # Scale Y to [0, 1] for training
    Y_scaled = (Y - 90) / 10  # Maps 90–100 to 0–1
    return X, F, Y_scaled, Y

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
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Constrain output to [0, 1]
        )
        self.residual = nn.Linear(d_model + 4, 1)

    def forward(self, x, features):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_projection(x) * np.sqrt(self.d_model)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = torch.cat((x, features), dim=1)
        out = self.fc(x) + torch.sigmoid(self.residual(x))  # Ensure residual is also in [0, 1]
        return out

def custom_mae_loss(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))  # Standard MAE

def train_model(model, train_loader, val_loader, device, num_epochs=100, patience=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Increased LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    os.makedirs('spo2_model_checkpoints', exist_ok=True)
    
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
        val_mae_rescaled = val_mae * 10  # Rescale MAE to SpO2 range (90–100)
        
        logging.info(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val SpO2 MAE: {val_mae_rescaled:.2f}%')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'spo2_model_checkpoints/best_spo2_hybrid_model.pth')
            logging.info(f'Model saved at epoch {epoch+1} with validation loss: {val_loss:.6f}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        scheduler.step()
        
        if epochs_no_improve >= patience:
            logging.info(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    return best_val_loss

def evaluate_model(model, test_loader, device, Y_original):
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
    y_true_rescaled = y_true * 10 + 90  # Rescale to 90–100
    y_pred_rescaled = y_pred * 10 + 90  # Rescale to 90–100
    mae = np.mean(np.abs(y_true_rescaled - y_pred_rescaled))
    
    avg_actual_spo2 = np.mean(y_true_rescaled)
    avg_predicted_spo2 = np.mean(y_pred_rescaled)
    logging.info(f'\n=== Test Results ===')
    logging.info(f'Test SpO2 MAE: {mae:.2f}%')
    logging.info(f'Average Actual SpO2: {avg_actual_spo2:.2f}%')
    logging.info(f'Average Predicted SpO2: {avg_predicted_spo2:.2f}%')
    logging.info(f'=== End of Test Results ===\n')
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(y_true_rescaled[:100], label='Actual SpO2 (%)', color='green')
    plt.plot(y_pred_rescaled[:100], label='Predicted SpO2 (%)', color='red', linestyle='--')
    plt.title('Actual vs Predicted SpO2 (First 100 Sequences)')
    plt.xlabel('Sequence Index')
    plt.ylabel('SpO2 (%)')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.scatter(y_true_rescaled, y_pred_rescaled, alpha=0.5)
    plt.plot([y_true_rescaled.min(), y_true_rescaled.max()], [y_true_rescaled.min(), y_true_rescaled.max()], 'r--', lw=2)
    plt.title(f'Predicted vs Actual SpO2 (MAE={mae:.2f}%)')
    plt.xlabel('Actual SpO2 (%)')
    plt.ylabel('Predicted SpO2 (%)')
    
    plt.subplot(3, 1, 3)
    errors = y_pred_rescaled - y_true_rescaled
    plt.hist(errors, bins=50, color='blue', alpha=0.7)
    plt.title('Error Distribution (Predicted - Actual SpO2)')
    plt.xlabel('Error (%)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Plot original SpO2 distribution
    plt.figure(figsize=(8, 5))
    plt.hist(Y_original, bins=50, color='purple', alpha=0.7)
    plt.title('Distribution of Original SpO2 Values')
    plt.xlabel('SpO2 (%)')
    plt.ylabel('Frequency')
    plt.show()
    
    return mae, avg_actual_spo2, avg_predicted_spo2

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    logging.info(f'Initial memory usage: {get_memory_usage():.1f} MB')
    
    window_size = 500
    X, F, Y_scaled, Y_original = load_and_preprocess_data(window_size=window_size)
    
    logging.info(f'Memory after data loading: {get_memory_usage():.1f} MB')
    cleanup_memory()
    
    logging.info(f'Data shapes: X={X.shape}, F={F.shape}, Y_scaled={Y_scaled.shape}')
    if X.shape[0] == 0 or Y_scaled.shape[0] == 0:
        logging.error("No valid data extracted!")
        return
    logging.info(f'X range: [{X.min():.3f}, {X.max():.3f}]')
    logging.info(f'Y_original range (SpO2 in %): [{Y_original.min():.3f}, {Y_original.max():.3f}]')
    logging.info(f'Y_original mean (SpO2 in %): {Y_original.mean():.3f}, Y_original std: {Y_original.std():.3f}')
    
    X_temp, X_test, F_temp, F_test, Y_temp, Y_test, Y_orig_temp, Y_orig_test = train_test_split(
        X, F, Y_scaled, Y_original, test_size=0.15, random_state=42)
    X_train, X_val, F_train, F_val, Y_train, Y_val, Y_orig_train, Y_orig_val = train_test_split(
        X_temp, F_temp, Y_temp, Y_orig_temp, test_size=0.15/0.85, random_state=42)
    logging.info(f'Dataset sizes: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}')
    
    X_train = torch.FloatTensor(X_train)
    F_train = torch.FloatTensor(F_train)
    Y_train = torch.FloatTensor(Y_train)
    X_val = torch.FloatTensor(X_val)
    F_val = torch.FloatTensor(F_val)
    Y_val = torch.FloatTensor(Y_val)
    X_test = torch.FloatTensor(X_test)
    F_test = torch.FloatTensor(F_test)
    Y_test = torch.FloatTensor(Y_test)
    
    logging.info(f'Memory after tensor conversion: {get_memory_usage():.1f} MB')
    cleanup_memory()
    
    batch_size = 8
    train_dataset = TensorDataset(X_train, F_train, Y_train)
    val_dataset = TensorDataset(X_val, F_val, Y_val)
    test_dataset = TensorDataset(X_test, F_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logging.info(f'Using batch size: {batch_size}')
    logging.info(f'Memory after DataLoader creation: {get_memory_usage():.1f} MB')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridTransformerCNN(
        input_size=2, 
        d_model=64,
        nhead=4,
        num_transformer_layers=2,
        num_cnn_filters=32,
        dropout=0.3
    ).to(device)
    
    logging.info(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    logging.info(f'Memory after model creation: {get_memory_usage():.1f} MB')
    
    logging.info('Starting training...')
    start_time = time.time()
    train_model(model, train_loader, val_loader, device, num_epochs=100, patience=30)
    logging.info(f'Training completed in {time.time() - start_time:.2f} seconds')
    
    logging.info('Loading best model from spo2_model_checkpoints/best_spo2_hybrid_model.pth')
    model.load_state_dict(torch.load('spo2_model_checkpoints/best_spo2_hybrid_model.pth'))
    
    logging.info('\nEvaluating on test set...')
    evaluate_model(model, test_loader, device, Y_orig_test)

if __name__ == "__main__":
    main()