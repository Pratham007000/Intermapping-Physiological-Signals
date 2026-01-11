import wfdb
import numpy as np
from scipy import signal
import pywt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import time
import psutil
import gc
import warnings
warnings.filterwarnings('ignore')

# 1. Preprocessing Functions
def denoise_emg(emg_signal, wavelet='db6', level=4):
    """Denoise EMG signal using wavelet transform"""
    coeffs = pywt.wavedec(emg_signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(emg_signal)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs, wavelet)
    return signal.medfilt(denoised, kernel_size=3)

def preprocess_emg(emg_signal, fs, lowcut=20, highcut=500):
    """Apply bandpass filter and rectification"""
    # Bandpass filter
    sos = signal.butter(4, [lowcut, highcut], btype='band', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, emg_signal)
    # Rectify (absolute value)
    rectified = np.abs(filtered)
    # Smooth with moving average
    window_size = int(fs * 0.1)  # 100ms window
    smoothed = np.convolve(rectified, np.ones(window_size)/window_size, mode='valid')
    return smoothed

def signal_quality_check(emg_signal, threshold=0.05):
    """Check EMG signal quality"""
    if np.std(emg_signal) < threshold or np.any(~np.isfinite(emg_signal)):
        return False
    # Check for excessive noise
    freqs, psd = signal.welch(emg_signal, fs=2000, nperseg=min(len(emg_signal), 2048))
    high_freq_power = np.sum(psd[freqs > 500]) / np.sum(psd)
    if high_freq_power > 0.6:
        return False
    return True

# 2. Feature Extraction
def extract_emg_features(emg_signal, fs, window_size=200):
    """Extract time and frequency domain features"""
    # Time-domain features
    rms = np.sqrt(np.mean(emg_signal**2))  # Root Mean Square
    mav = np.mean(np.abs(emg_signal))      # Mean Absolute Value
    zc = np.sum(np.diff(np.sign(emg_signal)) != 0) / len(emg_signal)  # Zero Crossings
    
    # Frequency-domain features
    freqs, psd = signal.welch(emg_signal, fs=fs, nperseg=min(len(emg_signal), window_size))
    mean_freq = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
    power = np.sum(psd)
    
    return np.array([rms, mav, zc, mean_freq, power])

# 3. Generate Synthetic EMG Data (since EMGDB is not available)
def generate_synthetic_emg_data(duration=30, fs=2000, window_size=2000):
    """Generate synthetic EMG data for demonstration"""
    try:
        print("Generating synthetic EMG data...")
        
        # Generate time vector
        t = np.linspace(0, duration, int(fs * duration))
        
        # Create synthetic EMG signal with multiple components
        # Base EMG activity (muscle contraction patterns)
        base_freq = 50  # Hz
        emg_signal = 0.5 * np.sin(2 * np.pi * base_freq * t)
        
        # Add multiple frequency components typical of EMG
        for freq in [80, 120, 200, 300]:
            amplitude = 0.3 / (freq / 50)  # Lower amplitude for higher frequencies
            emg_signal += amplitude * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi))
        
        # Add realistic EMG burst patterns
        burst_locations = np.random.choice(len(t), size=int(len(t) * 0.1), replace=False)
        for loc in burst_locations:
            burst_duration = int(0.1 * fs)  # 100ms bursts
            start_idx = max(0, loc - burst_duration // 2)
            end_idx = min(len(emg_signal), loc + burst_duration // 2)
            burst_envelope = np.exp(-((np.arange(end_idx - start_idx) - (end_idx - start_idx) // 2)**2) / (2 * (burst_duration // 4)**2))
            emg_signal[start_idx:end_idx] += 2.0 * burst_envelope
        
        # Add physiological noise
        noise = np.random.normal(0, 0.1, len(emg_signal))
        emg_signal += noise
        
        # Ensure signal quality
        if not signal_quality_check(emg_signal, threshold=0.01):
            print("Generated signal quality check failed, regenerating...")
            return generate_synthetic_emg_data(duration, fs, window_size)
        
        # Process the signal
        emg_denoised = denoise_emg(emg_signal, level=6)
        emg_processed = preprocess_emg(emg_denoised, fs)
        
        # Create windows
        step_size = window_size // 2
        num_samples = (len(emg_processed) - window_size) // step_size + 1
        X = np.zeros((num_samples, window_size))
        F = np.zeros((num_samples, 5))  # 5 features
        Y = np.zeros((num_samples, window_size))  # Ground truth EMG
        
        for i in range(num_samples):
            start = i * step_size
            end = start + window_size
            if end <= len(emg_processed) and end <= len(emg_denoised):
                window = emg_processed[start:end]
                if len(window) == window_size:
                    X[i] = window
                    F[i] = extract_emg_features(window, fs, window_size)
                    Y[i] = emg_denoised[start:end]  # Ground truth is denoised raw EMG
        
        # Remove any incomplete windows
        valid_idx = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
        valid_idx = valid_idx & (np.sum(X, axis=1) != 0) & (np.sum(Y, axis=1) != 0)
        
        print(f"Generated {np.sum(valid_idx)} valid EMG windows")
        return X[valid_idx], F[valid_idx], Y[valid_idx], fs
    
    except Exception as e:
        print(f"Error generating synthetic EMG data: {e}")
        return None, None, None, None

# Alternative: Load and Preprocess BIDMC Data as EMG substitute
def load_bidmc_as_emg_data(record_name='bidmc/bidmc01', window_size=2000):
    """Load BIDMC data and treat ECG as EMG for demonstration"""
    try:
        print(f"Loading BIDMC record: {record_name} (treating as EMG)")
        record = wfdb.rdrecord(record_name, pn_dir='bidmc')
        fs = record.fs
        # Use ECG signal and process it as if it were EMG
        emg_signal = record.p_signal[:, 0]  # ECG channel as EMG substitute
        
        # Scale and modify to be more EMG-like
        emg_signal = np.abs(emg_signal) * 10  # Make it more EMG-like
        
        # Denoise and quality check
        if not signal_quality_check(emg_signal, threshold=0.01):
            raise ValueError("Poor signal quality")
        
        emg_denoised = denoise_emg(emg_signal, level=int(np.log2(len(emg_signal))-3))
        emg_processed = preprocess_emg(emg_denoised, fs)
        
        # Create windows
        step_size = window_size // 2
        num_samples = (len(emg_processed) - window_size) // step_size + 1
        X = np.zeros((num_samples, window_size))
        F = np.zeros((num_samples, 5))  # 5 features
        Y = np.zeros((num_samples, window_size))  # Ground truth EMG
        
        for i in range(num_samples):
            start = i * step_size
            end = start + window_size
            if end <= len(emg_processed) and end <= len(emg_denoised):
                window = emg_processed[start:end]
                if len(window) == window_size:
                    X[i] = window
                    F[i] = extract_emg_features(window, fs, window_size)
                    Y[i] = emg_denoised[start:end]  # Ground truth is denoised raw EMG
        
        valid_idx = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
        return X[valid_idx], F[valid_idx], Y[valid_idx], fs
    
    except Exception as e:
        print(f"Error loading {record_name}: {e}")
        return None, None, None, None

# 4. Simple EMG Prediction Model
class EMGNet(nn.Module):
    """Simple neural network for EMG prediction"""
    def __init__(self, input_size, feature_size=5, hidden_size=128):
        super(EMGNet, self).__init__()
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3)
        )
        self.signal_processor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.combined = nn.Sequential(
            nn.Linear(hidden_size + 64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
    
    def forward(self, x, features):
        # Process features
        feat_out = self.feature_processor(features)
        # Process signal
        x = x.unsqueeze(1)  # Add channel dimension
        signal_out = self.signal_processor(x).mean(dim=2)
        # Combine
        combined = torch.cat((feat_out, signal_out), dim=1)
        return self.combined(combined)

# 5. Training Function
def train_emg_model(model, train_loader, val_loader, device, num_epochs=50, patience=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    os.makedirs('emg_model_checkpoints', exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, F_batch, y_batch in train_loader:
            X_batch, F_batch, y_batch = X_batch.to(device), F_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch, F_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, F_batch, y_batch in val_loader:
                X_batch, F_batch, y_batch = X_batch.to(device), F_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch, F_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'emg_model_checkpoints/best_emg_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    return best_val_loss

# 6. Evaluation and Comparison
def evaluate_and_compare(model, test_loader, device, fs):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, F_batch, y_batch in test_loader:
            X_batch, F_batch = X_batch.to(device), F_batch.to(device)
            pred = model(X_batch, F_batch)
            y_true.append(y_batch.numpy())
            y_pred.append(pred.cpu().numpy())
    
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0,1]
    
    # Print results
    print(f'\n=== EMG Comparison Results ===')
    print(f'Mean Absolute Error (MAE): {mae:.6f}')
    print(f'Root Mean Square Error (RMSE): {rmse:.6f}')
    print(f'Correlation Coefficient: {corr:.3f}')
    print(f'=== End of Results ===\n')
    
    # Visualization
    plt.figure(figsize=(12, 10))
    
    # Time domain comparison
    plt.subplot(2, 1, 1)
    n_samples = min(2000, len(y_true[0]))
    t = np.arange(n_samples) / fs
    plt.plot(t, y_true[0][:n_samples], label='Ground Truth EMG', color='blue')
    plt.plot(t, y_pred[0][:n_samples], label='Predicted EMG', color='red', linestyle='--')
    plt.title('Ground Truth vs Predicted EMG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # Error distribution
    plt.subplot(2, 1, 2)
    errors = y_pred.flatten() - y_true.flatten()
    plt.hist(errors, bins=50, color='purple', alpha=0.7)
    plt.title(f'Error Distribution (MAE={mae:.6f}, RMSE={rmse:.6f})')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return mae, rmse, corr

# 7. Main Function
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    print(f'Initial memory usage: {get_memory_usage():.1f} MB')
    
    # Load data - try synthetic EMG data since EMGDB is not available
    window_size = 2000  # 1s at 2000Hz
    print("Attempting to generate synthetic EMG data...")
    X, F, Y, fs = generate_synthetic_emg_data(duration=60, fs=2000, window_size=window_size)
    
    if X is None:
        print("Failed to generate synthetic data, trying BIDMC as EMG substitute...")
        X, F, Y, fs = load_bidmc_as_emg_data(record_name='bidmc01', window_size=window_size)
    
    if X is None:
        print("Failed to load any data for EMG analysis")
        return
    
    print(f'Memory after data loading: {get_memory_usage():.1f} MB')
    cleanup_memory()
    
    # Data stats
    print(f'Data shapes: X={X.shape}, F={F.shape}, Y={Y.shape}')
    print(f'X range: [{X.min():.3f}, {X.max():.3f}]')
    print(f'Y range: [{Y.min():.3f}, {Y.max():.3f}]')
    
    # Split dataset
    X_temp, X_test, F_temp, F_test, Y_temp, Y_test = train_test_split(X, F, Y, test_size=0.15, random_state=42)
    X_train, X_val, F_train, F_val, Y_train, Y_val = train_test_split(X_temp, F_temp, Y_temp, test_size=0.15/0.85, random_state=42)
    print(f'Dataset sizes: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}')
    
    # Convert to tensors
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
    
    print(f'Memory after DataLoader creation: {get_memory_usage():.1f} MB')
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EMGNet(input_size=window_size, feature_size=5, hidden_size=128).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Train model
    print('Starting training...')
    start_time = time.time()
    train_emg_model(model, train_loader, val_loader, device)
    print(f'Training completed in {time.time() - start_time:.2f} seconds')
    
    # Load best model
    model.load_state_dict(torch.load('emg_model_checkpoints/best_emg_model.pth'))
    
    # Evaluate and compare
    print('\nComparing predicted EMG with ground truth...')
    evaluate_and_compare(model, test_loader, device, fs)

if __name__ == "__main__":
    main()