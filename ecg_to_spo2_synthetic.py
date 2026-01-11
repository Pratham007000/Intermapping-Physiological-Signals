import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import time
import psutil
import gc
import warnings
import logging
from scipy import signal
from scipy.signal import find_peaks, butter, sosfilt
from scipy.ndimage import gaussian_filter1d
try:
    import pywt
except ImportError:
    pywt = None
    logging.warning("PyWavelets not available. Wavelet denoising will be disabled.")

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Advanced Preprocessing
def denoise_signal(signal_data, wavelet='db4', level=4):
    if pywt is None:
        logging.warning("PyWavelets not available, using median filter only")
        return signal.medfilt(signal_data, kernel_size=5)
    
    try:
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        denoised = pywt.waverec(coeffs, wavelet)
        return signal.medfilt(denoised, kernel_size=5)
    except Exception as e:
        logging.warning(f"Wavelet denoising failed: {e}, using median filter only")
        return signal.medfilt(signal_data, kernel_size=5)

def signal_quality_check(signal_data, threshold=0.05, debug=False):
    # Check for basic signal validity
    if np.std(signal_data) < threshold or np.any(~np.isfinite(signal_data)):
        if debug:
            logging.info(f"Signal rejected: std={np.std(signal_data):.4f} < {threshold} or non-finite values={np.any(~np.isfinite(signal_data))}")
        return False
    
    # Much more relaxed quality check for ECG data
    # ECG has significant high-frequency content that's normal
    freqs, psd = signal.welch(signal_data, fs=250, nperseg=min(len(signal_data), 256))
    
    # Check for reasonable frequency distribution (less strict)
    # Look for physiological frequency content (0.5-40 Hz is normal for ECG)
    physio_band = (freqs >= 0.5) & (freqs <= 40.0)
    physio_power = np.sum(psd[physio_band]) / np.sum(psd) if np.any(physio_band) else 0
    
    if debug:
        logging.info(f"Signal stats: std={np.std(signal_data):.4f}, physio_power={physio_power:.4f}")
        logging.info(f"Signal range: [{signal_data.min():.4f}, {signal_data.max():.4f}]")
    
    # Much more lenient threshold - require at least 20% physiological content
    if physio_power < 0.2:
        if debug:
            logging.warning(f"Signal rejected: insufficient physiological content ({physio_power:.4f} < 0.2)")
        return False
    
    return True

def generate_synthetic_ppg(ecg_data, fs=250, delay=58, sigma=2, scale_factor=1.5):
    # Improved PPG synthesis with multiple physiological components
    peaks, _ = find_peaks(ecg_data, height=0.25, distance=20)
    ppg = np.zeros_like(ecg_data)
    
    # Primary pulse component
    for peak in peaks:
        if peak + delay < len(ecg_data):
            ppg[peak + delay] = ecg_data[peak] * scale_factor
    
    # Add secondary pulse components (dicrotic notch simulation)
    for peak in peaks:
        secondary_delay = delay + int(fs * 0.15)  # ~150ms after primary peak
        if peak + secondary_delay < len(ecg_data):
            ppg[peak + secondary_delay] += ecg_data[peak] * scale_factor * 0.3
    
    # Add respiratory modulation (realistic physiological variation)
    resp_freq = 0.25  # 15 breaths per minute
    t = np.arange(len(ecg_data)) / fs
    respiratory_modulation = 0.1 * np.sin(2 * np.pi * resp_freq * t)
    
    ppg = gaussian_filter1d(ppg, sigma=sigma)
    ppg = np.roll(ppg, delay)
    ppg[:delay] = ppg[delay]
    
    # Apply respiratory modulation
    ppg *= (1 + respiratory_modulation)
    
    # Add realistic noise
    noise_level = 0.05 * np.std(ppg)
    ppg += np.random.normal(0, noise_level, len(ppg))
    
    ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
    return ppg

def extract_spo2(ppg, fs=250):
    # Advanced SpO2 extraction with improved physiological modeling
    if len(ppg) < fs // 2:
        return 0
    
    # Improved bandpass filter for PPG heart rate range
    sos = signal.butter(3, [0.5, 8.0], btype='band', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, ppg)
    filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    
    # Advanced peak detection with adaptive thresholds
    peaks, properties = signal.find_peaks(filtered, 
                                        distance=int(fs*0.4),  # Min 150 BPM
                                        prominence=0.1,
                                        width=int(fs*0.05))  # Min width 50ms
    
    troughs, _ = signal.find_peaks(-filtered, 
                                 distance=int(fs*0.4), 
                                 prominence=0.1)
    
    # Calculate multiple physiological features
    features = {}
    
    # Heart rate variability from peaks
    if len(peaks) >= 3:
        rr_intervals = np.diff(peaks) / fs  # R-R intervals in seconds
        features['hr_mean'] = 60.0 / np.mean(rr_intervals)  # Mean heart rate
        features['hr_std'] = np.std(rr_intervals)  # HRV
    else:
        features['hr_mean'] = 70.0  # Default
        features['hr_std'] = 0.05
    
    # Pulse amplitude and regularity
    if len(peaks) >= 2 and len(troughs) >= 2:
        # AC component (pulse amplitude)
        peak_amplitudes = filtered[peaks]
        trough_amplitudes = filtered[troughs]
        
        # Match peaks and troughs for proper AC calculation
        min_len = min(len(peak_amplitudes), len(trough_amplitudes))
        ac_component = np.mean(peak_amplitudes[:min_len] - trough_amplitudes[:min_len])
        
        # Pulse regularity (consistency of pulse shape)
        features['pulse_regularity'] = 1.0 / (1.0 + np.std(peak_amplitudes))
        features['ac_amplitude'] = ac_component
    else:
        features['pulse_regularity'] = 0.5
        features['ac_amplitude'] = np.std(filtered)
    
    # DC component (baseline)
    features['dc_component'] = np.mean(np.abs(filtered))
    
    # Frequency domain analysis
    freqs, psd = signal.welch(filtered, fs=fs, nperseg=min(len(filtered), fs*2))
    
    # Heart rate frequency band power (0.8-3.5 Hz corresponds to 48-210 BPM)
    hr_band = (freqs >= 0.8) & (freqs <= 3.5)
    hr_power = np.sum(psd[hr_band]) / (np.sum(psd) + 1e-8) if np.any(hr_band) else 0.5
    features['hr_power'] = hr_power
    
    # Respiratory frequency band (0.15-0.4 Hz corresponds to 9-24 breaths/min)
    resp_band = (freqs >= 0.15) & (freqs <= 0.4)
    resp_power = np.sum(psd[resp_band]) / (np.sum(psd) + 1e-8) if np.any(resp_band) else 0.1
    features['resp_power'] = resp_power
    
    # Advanced SpO2 calculation using multiple physiological relationships
    
    # 1. AC/DC ratio based SpO2 (primary method)
    if features['dc_component'] > 0:
        ac_dc_ratio = features['ac_amplitude'] / features['dc_component']
        # Empirical relationship for SpO2 from AC/DC ratio
        spo2_ac_dc = 100 - 25 * np.clip(ac_dc_ratio, 0, 0.4)
    else:
        spo2_ac_dc = 95.0
    
    # 2. Heart rate based adjustment (physiological correlation)
    # Normal resting HR (60-100 BPM) correlates with better perfusion
    hr_factor = 1.0
    if features['hr_mean'] < 60:  # Bradycardia
        hr_factor = 0.98
    elif features['hr_mean'] > 100:  # Tachycardia  
        hr_factor = 0.97
    
    # 3. Signal quality based adjustment
    signal_quality = features['pulse_regularity'] * features['hr_power']
    quality_factor = 0.95 + 0.05 * signal_quality  # 0.95-1.0 range
    
    # 4. Respiratory modulation effect
    resp_factor = 1.0 - 0.02 * features['resp_power']  # Strong respiratory modulation slightly reduces SpO2
    
    # Combine all factors for final SpO2
    base_spo2 = spo2_ac_dc * hr_factor * quality_factor * resp_factor
    
    # Add physiologically realistic random variation
    # SpO2 naturally varies ±1-2% due to breathing, movement, etc.
    normal_variation = np.random.normal(0, 1.2)
    base_spo2 += normal_variation
    
    # Apply realistic physiological constraints
    # Healthy individuals: 95-100%, some pathological cases: 90-95%
    if base_spo2 > 98:
        final_spo2 = np.clip(base_spo2, 96, 100)  # Healthy range
    else:
        final_spo2 = np.clip(base_spo2, 90, 98)   # Lower but realistic range
    
    return final_spo2

def extract_features(ecg, ppg, fs=250):
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
    
    if pywt is not None:
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
            logging.warning(f"CWT error: {e}, using fallback")
            cwt_feature = 0
    else:
        # Fallback when pywt is not available - use simple variance
        cwt_feature = np.var(ppg)
    
    return np.array([rsa_feature, ac_dc_ratio, freq_feature, cwt_feature])

def load_and_preprocess_data(window_size=500):
    try:
        ecg_data = pd.read_csv("ecg_data_20250701_172937.csv")["ECG Amplitude"].values
        ecg_data = ecg_data.astype(np.float32)
    except FileNotFoundError:
        logging.error("Error: ecg_data_20250701_172937.csv not found. Please ensure the file is in the current directory.")
        exit()
    except KeyError:
        logging.error("Error: CSV file must contain an 'ECG Amplitude' column. Check the file structure.")
        exit()

    ecg_data = (ecg_data - np.mean(ecg_data)) / np.std(ecg_data)
    fs = 250  # Assuming 250 Hz sampling rate as per previous context
    num_samples = (len(ecg_data) - window_size) // (window_size // 2) + 1
    X = np.zeros((num_samples, window_size, 2))
    F = np.zeros((num_samples, 4))
    Y = np.zeros(num_samples)

    debug_first_few = 3  # Debug first 3 windows
    
    for i in range(num_samples):
        start = i * (window_size // 2)
        end = start + window_size
        if end > len(ecg_data):
            break
            
        ecg_window = ecg_data[start:end]
        
        # Debug first few windows
        if i < debug_first_few:
            logging.info(f"\n=== Debug Window {i} ===")
            logging.info(f"ECG window stats: mean={ecg_window.mean():.4f}, std={ecg_window.std():.4f}")
            logging.info(f"ECG window range: [{ecg_window.min():.4f}, {ecg_window.max():.4f}]")
            
            # Check if ECG passes quality check
            ecg_quality = signal_quality_check(ecg_window, debug=True)
            logging.info(f"ECG quality check: {ecg_quality}")
        
        ppg_window = generate_synthetic_ppg(ecg_window, fs=fs)
        
        if i < debug_first_few:
            logging.info(f"PPG window stats: mean={ppg_window.mean():.4f}, std={ppg_window.std():.4f}")
            logging.info(f"PPG window range: [{ppg_window.min():.4f}, {ppg_window.max():.4f}]")
            
            # Check if PPG passes quality check
            ppg_quality = signal_quality_check(ppg_window, debug=True)
            logging.info(f"PPG quality check: {ppg_quality}")
        
        X[i, :, 0] = ecg_window
        X[i, :, 1] = ppg_window
        F[i] = extract_features(ecg_window, ppg_window, fs)
        Y[i] = extract_spo2(ppg_window, fs)
        
        if i < debug_first_few:
            logging.info(f"SpO2 value: {Y[i]:.2f}")
            logging.info(f"Features: {F[i]}")

    valid_idx = (Y >= 90) & (Y <= 100)
    logging.info(f"Found {np.sum(valid_idx)} valid windows out of {num_samples}")
    if np.sum(valid_idx) > 0:
        X = X[valid_idx]
        F = F[valid_idx]
        Y = Y[valid_idx]
    else:
        raise Exception("No valid SpO2 data extracted!")

    Y_scaled = (Y - 90) / 10  # Maps 90–100 to 0–1
    return X, F, Y_scaled, Y

class AdvancedSpO2Net(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_transformer_layers=4, num_cnn_filters=64, dropout=0.2):
        super(AdvancedSpO2Net, self).__init__()
        self.d_model = d_model
        
        # Multi-scale CNN feature extraction
        self.multi_scale_cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_size, num_cnn_filters//4, kernel_size=3, padding=1),
                nn.BatchNorm1d(num_cnn_filters//4),
                nn.ReLU(),
                nn.Conv1d(num_cnn_filters//4, num_cnn_filters//4, kernel_size=3, padding=1),
                nn.BatchNorm1d(num_cnn_filters//4),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(input_size, num_cnn_filters//4, kernel_size=7, padding=3),
                nn.BatchNorm1d(num_cnn_filters//4),
                nn.ReLU(),
                nn.Conv1d(num_cnn_filters//4, num_cnn_filters//4, kernel_size=7, padding=3),
                nn.BatchNorm1d(num_cnn_filters//4),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(input_size, num_cnn_filters//4, kernel_size=15, padding=7),
                nn.BatchNorm1d(num_cnn_filters//4),
                nn.ReLU(),
                nn.Conv1d(num_cnn_filters//4, num_cnn_filters//4, kernel_size=15, padding=7),
                nn.BatchNorm1d(num_cnn_filters//4),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(input_size, num_cnn_filters//4, kernel_size=31, padding=15),
                nn.BatchNorm1d(num_cnn_filters//4),
                nn.ReLU(),
                nn.Conv1d(num_cnn_filters//4, num_cnn_filters//4, kernel_size=31, padding=15),
                nn.BatchNorm1d(num_cnn_filters//4),
                nn.ReLU(),
            )
        ])
        
        # Attention-based feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(num_cnn_filters, num_cnn_filters, kernel_size=1),
            nn.BatchNorm1d(num_cnn_filters),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_cnn_filters, num_cnn_filters//4),
            nn.ReLU(),
            nn.Linear(num_cnn_filters//4, num_cnn_filters),
            nn.Sigmoid()
        )
        
        # Positional encoding for transformer
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 500, d_model) * 0.1, requires_grad=True
        )
        
        # Input projection for transformer
        self.input_projection = nn.Sequential(
            nn.Linear(num_cnn_filters, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Advanced transformer with residual connections
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=dropout, 
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Feature fusion network
        self.feature_processor = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Multi-head attention for final fusion (removed due to dimension issues)
        # self.cross_attention = nn.MultiheadAttention(
        #     embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        # )
        
        # Advanced prediction head with skip connections
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model + 32, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.LayerNorm(d_model//2),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(d_model//2, d_model//4),
            nn.ReLU(),
            nn.Linear(d_model//4, 1),
            nn.Sigmoid()
        )
        
        # Residual connection for direct feature influence
        self.direct_prediction = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Learnable fusion weight
        self.fusion_weight = nn.Parameter(torch.tensor(0.8))

    def forward(self, x, features):
        batch_size, seq_len, _ = x.size()
        
        # Multi-scale CNN feature extraction
        x_transposed = x.transpose(1, 2)  # (batch, channels, seq_len)
        
        multi_scale_features = []
        for cnn in self.multi_scale_cnn:
            multi_scale_features.append(cnn(x_transposed))
        
        # Concatenate multi-scale features
        cnn_features = torch.cat(multi_scale_features, dim=1)  # (batch, total_filters, seq_len)
        
        # Attention-based feature fusion
        attention_weights = self.feature_fusion(cnn_features)
        attention_weights = attention_weights.unsqueeze(-1)  # (batch, filters, 1)
        cnn_features = cnn_features * attention_weights
        
        # Transpose back for transformer
        cnn_features = cnn_features.transpose(1, 2)  # (batch, seq_len, filters)
        
        # Project to transformer dimension
        transformer_input = self.input_projection(cnn_features)
        
        # Add positional encoding
        if transformer_input.size(1) <= self.positional_encoding.size(1):
            pos_encoding = self.positional_encoding[:, :transformer_input.size(1), :]
            transformer_input = transformer_input + pos_encoding
        
        # Transformer processing
        transformer_output = self.transformer(transformer_input)
        
        # Global average pooling with attention
        attention_weights_seq = torch.softmax(
            torch.sum(transformer_output, dim=-1), dim=1
        ).unsqueeze(-1)
        
        global_features = torch.sum(
            transformer_output * attention_weights_seq, dim=1
        )
        
        # Process auxiliary features
        processed_features = self.feature_processor(features)
        
        # Simple feature fusion (replace complex cross-attention)
        # Combine global features with processed auxiliary features
        combined_features = torch.cat([global_features, processed_features], dim=1)
        
        # Main prediction
        main_prediction = self.prediction_head(combined_features)
        
        # Direct feature prediction (residual)
        direct_prediction = self.direct_prediction(processed_features)
        
        # Learnable fusion of predictions
        final_prediction = (
            self.fusion_weight * main_prediction + 
            (1 - self.fusion_weight) * direct_prediction
        )
        
        return final_prediction

def custom_mae_loss(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))  # Standard MAE

def train_model(model, train_loader, val_loader, device, num_epochs=150, patience=40):
    # Advanced optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    
    # Advanced learning rate scheduling
    warmup_epochs = 10
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-6)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    os.makedirs('spo2_model_checkpoints', exist_ok=True)
    
    # Advanced loss function combining MAE and MSE
    def advanced_loss(y_pred, y_true):
        mae = torch.mean(torch.abs(y_pred - y_true))
        mse = torch.mean((y_pred - y_true) ** 2)
        # Weighted combination - emphasize both absolute and squared errors
        return 0.7 * mae + 0.3 * mse
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_mae = 0
        num_batches = 0
        
        for X_batch, F_batch, y_batch in train_loader:
            X_batch, F_batch, y_batch = X_batch.to(device), F_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch, F_batch).squeeze()
            
            # Advanced loss calculation
            loss = advanced_loss(y_pred, y_batch)
            
            # L2 regularization on model parameters
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            loss += 1e-6 * l2_reg
            
            loss.backward()
            
            # Gradient clipping with adaptive norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            train_mae += torch.abs(y_pred - y_batch).sum().item()
            num_batches += 1
        
        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for X_batch, F_batch, y_batch in val_loader:
                X_batch, F_batch, y_batch = X_batch.to(device), F_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch, F_batch).squeeze()
                loss = advanced_loss(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_mae += torch.abs(y_pred - y_batch).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        
        # Rescale MAE to SpO2 percentage range (90-100%)
        train_mae_rescaled = train_mae * 10
        val_mae_rescaled = val_mae * 10
        
        # Advanced learning rate scheduling
        if epoch < warmup_epochs:
            scheduler1.step()
        else:
            scheduler2.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 5 == 0 or epoch < 10:
            logging.info(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Train MAE: {train_mae_rescaled:.2f}%, '
                        f'Val Loss: {val_loss:.6f}, Val MAE: {val_mae_rescaled:.2f}%, LR: {current_lr:.2e}')
        
        # Model checkpointing with improved criteria
        improvement_threshold = 0.001  # Require at least 0.1% improvement
        if val_loss < best_val_loss - improvement_threshold:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae_rescaled
            }, 'spo2_model_checkpoints/best_spo2_hybrid_model.pth')
            logging.info(f'Model saved at epoch {epoch+1} with validation loss: {val_loss:.6f}, MAE: {val_mae_rescaled:.2f}%')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Early stopping with validation loss plateau detection
        if epochs_no_improve >= patience:
            logging.info(f'Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)')
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
    model = AdvancedSpO2Net(
        input_size=2, 
        d_model=128,  # Increased model capacity
        nhead=8,      # More attention heads
        num_transformer_layers=6,  # Deeper transformer
        num_cnn_filters=128,  # More CNN filters
        dropout=0.15  # Reduced dropout for better learning
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