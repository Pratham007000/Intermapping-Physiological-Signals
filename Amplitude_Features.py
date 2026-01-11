import wfdb
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

# 1. Preprocessing
def denoise_signal(signal_data, fs=125):
    """Denoise PPG signal using a band-pass filter."""
    sos = signal.butter(4, [0.5, 5.0], btype='band', fs=fs, output='sos')
    denoised = signal.sosfilt(sos, signal_data)
    return denoised

def extract_ppg_peaks(ppg_signal, fs=125, min_distance=0.4):
    """Extract peaks from PPG signal and their amplitudes."""
    ppg_signal = denoise_signal(ppg_signal)
    peaks, properties = signal.find_peaks(ppg_signal, distance=int(min_distance * fs), prominence=0.1)
    amplitudes = ppg_signal[peaks]
    return peaks, amplitudes

def extract_features(ppg_signal, fs=125, window_size=500):
    """Extract statistical features from PPG signal windows."""
    # Basic statistical features
    mean_val = np.mean(ppg_signal)
    std_val = np.std(ppg_signal)
    skewness = np.mean((ppg_signal - mean_val) ** 3) / (std_val ** 3 + 1e-8)
    kurtosis = np.mean((ppg_signal - mean_val) ** 4) / (std_val ** 4 + 1e-8)
    
    # Frequency-domain features
    freqs, psd = signal.welch(ppg_signal, fs=fs, nperseg=min(len(ppg_signal), fs*2))
    dominant_freq = freqs[np.argmax(psd)]
    power = np.sum(psd)
    
    return np.array([mean_val, std_val, skewness, kurtosis, dominant_freq, power])

def load_and_preprocess_data(window_size=500):
    """Load BIDMC dataset and preprocess for amplitude prediction."""
    records = ['bidmc01', 'bidmc02', 'bidmc03', 'bidmc04', 'bidmc05']
    X_all, F_all, Y_all = [], [], []
    fs = 125  # Sampling frequency from BIDMC dataset
    
    for record_name in records:
        try:
            print(f"Loading record: {record_name}")
            record = wfdb.rdrecord(record_name, pn_dir='bidmc/1.0.0/')
            ppg = record.p_signal[:, 1]  # PPG signal
            
            # Normalize PPG
            ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
            
            # Extract peaks and amplitudes for the entire signal
            peaks, amplitudes = extract_ppg_peaks(ppg, fs)
            
            # Create overlapping windows
            step_size = window_size // 2
            num_samples = (len(ppg) - window_size) // step_size + 1
            X = np.zeros((num_samples, window_size))
            F = np.zeros((num_samples, 6))  # 6 features
            Y = np.zeros(num_samples)
            
            for i in range(num_samples):
                start = i * step_size
                end = start + window_size
                window = ppg[start:end]
                
                # Extract features
                F[i] = extract_features(window, fs)
                
                # Find peaks in this window
                window_peaks, _ = signal.find_peaks(window, distance=int(0.4 * fs), prominence=0.1)
                if len(window_peaks) > 0:
                    # Use mean amplitude of peaks in the window
                    window_amplitudes = window[window_peaks]
                    Y[i] = np.mean(window_amplitudes) if len(window_amplitudes) > 0 else 0
                else:
                    Y[i] = 0
                
                X[i] = window
            
            valid_idx = Y != 0  # Filter out windows with no valid amplitude
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
    return X, F, Y, fs

# 2. Neural Network Model
class AmplitudePredictor(nn.Module):
    def __init__(self, input_size=1, feature_size=6, hidden_size=64, dropout=0.3):
        super(AmplitudePredictor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 500 + feature_size, hidden_size),  # Adjust based on window_size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x, features):
        x = self.conv(x)  # [batch, channels, seq_len]
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat((x, features), dim=1)  # Concatenate with features
        return self.fc(x).squeeze()

# 3. Training Loop
def train_model(model, train_loader, val_loader, device, num_epochs=50, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.L1Loss()  # MAE loss
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    os.makedirs('amplitude_model_checkpoints', exist_ok=True)
    
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'amplitude_model_checkpoints/best_amplitude_model.pth')
            print(f'Model saved at epoch {epoch+1} with validation loss: {val_loss:.6f}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        scheduler.step(val_loss)
        
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    return best_val_loss

# 4. Evaluation and Visualization
def evaluate_model(model, test_loader, device, fs=125):
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
    mae = np.mean(np.abs(y_true - y_pred))
    
    print(f'\n=== Test Results ===')
    print(f'Test Amplitude MAE: {mae:.4f}')
    print(f'Average Actual Amplitude: {np.mean(y_true):.4f}')
    print(f'Average Predicted Amplitude: {np.mean(y_pred):.4f}')
    print(f'=== End of Test Results ===\n')
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(y_true[:100], label='Actual Amplitude', color='blue')
    plt.plot(y_pred[:100], label='Predicted Amplitude', color='red', linestyle='--')
    plt.title('Actual vs Predicted PPG Amplitude (First 100 Windows)')
    plt.xlabel('Window Index')
    plt.ylabel('Amplitude (Normalized)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f'Predicted vs Actual Amplitude (MAE={mae:.4f})')
    plt.xlabel('Actual Amplitude (Normalized)')
    plt.ylabel('Predicted Amplitude (Normalized)')
    
    plt.tight_layout()
    plt.show()
    
    return mae

# 5. Memory Management
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 6. Main Function
def main():
    print(f'Initial memory usage: {get_memory_usage():.1f} MB')
    
    # Load and preprocess data
    window_size = 500
    X, F, Y, fs = load_and_preprocess_data(window_size=window_size)
    
    print(f'Memory after data loading: {get_memory_usage():.1f} MB')
    cleanup_memory()
    
    # Data stats
    print(f'Data shapes: X={X.shape}, F={F.shape}, Y={Y.shape}')
    print(f'Y range (Amplitude): [{Y.min():.3f}, {Y.max():.3f}]')
    print(f'Y mean (Amplitude): {Y.mean():.3f}, Y std: {Y.std():.3f}')
    
    # Split dataset
    X_temp, X_test, F_temp, F_test, Y_temp, Y_test = train_test_split(X, F, Y, test_size=0.15, random_state=42)
    X_train, X_val, F_train, F_val, Y_train, Y_val = train_test_split(X_temp, F_temp, Y_temp, test_size=0.15/0.85, random_state=42)
    print(f'Dataset sizes: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}')
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
    F_train = torch.FloatTensor(F_train)
    Y_train = torch.FloatTensor(Y_train)
    X_val = torch.FloatTensor(X_val).unsqueeze(1)
    F_val = torch.FloatTensor(F_val)
    Y_val = torch.FloatTensor(Y_val)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)
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
    model = AmplitudePredictor(input_size=1, feature_size=6, hidden_size=64).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Memory after model creation: {get_memory_usage():.1f} MB')
    
    # Train model
    print('Starting training...')
    start_time = time.time()
    train_model(model, train_loader, val_loader, device, num_epochs=50, patience=10)
    print(f'Training completed in {time.time() - start_time:.2f} seconds')
    
    # Load best model
    print('Loading best model from amplitude_model_checkpoints/best_amplitude_model.pth')
    model.load_state_dict(torch.load('amplitude_model_checkpoints/best_amplitude_model.pth'))
    
    # Evaluate on test set
    print('\nEvaluating on test set...')
    evaluate_model(model, test_loader, device, fs)

if __name__ == "__main__":
    main()