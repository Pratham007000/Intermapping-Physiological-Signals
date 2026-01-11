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

# 1. Preprocessing Functions
def denoise_signal(signal_data, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs, wavelet)
    return signal.medfilt(denoised, kernel_size=5)

def extract_beat_intervals(signal_data, fs, signal_type='ecg', min_distance=0.4):
    signal_data = denoise_signal(signal_data)
    prominence = np.std(signal_data) * 0.5
    peaks, _ = signal.find_peaks(signal_data, distance=int(fs * min_distance), prominence=prominence)
    
    if len(peaks) < 2:
        return np.array([])
    
    intervals = np.diff(peaks) / fs  # Convert to seconds
    valid_intervals = intervals[(intervals >= 0.3) & (intervals <= 2.0)]  # 30-200 bpm
    return valid_intervals if len(valid_intervals) > 0 else np.array([])

def prepare_sequences(intervals, seq_length=10):
    if len(intervals) < seq_length + 1:
        return np.array([]), np.array([])
    
    X, y = [], []
    for i in range(len(intervals) - seq_length):
        X.append(intervals[i:i + seq_length])
        y.append(intervals[i + seq_length])
    return np.array(X), np.array(y)

def load_and_preprocess_data(seq_length=10, window_size=1000):
    records = ['bidmc01', 'bidmc02', 'bidmc03', 'bidmc04', 'bidmc05']
    X_all, y_all = [], []
    
    for record_name in records:
        try:
            print(f"Loading record: {record_name}")
            record = wfdb.rdrecord(record_name, pn_dir='bidmc/1.0.0/')
            fs = record.fs
            ecg = record.p_signal[:, 0]
            ppg = record.p_signal[:, 1]
            
            # Normalize signals
            ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
            ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
            
            # Extract intervals from both ECG and PPG
            ecg_intervals = extract_beat_intervals(ecg, fs, signal_type='ecg')
            ppg_intervals = extract_beat_intervals(ppg, fs, signal_type='ppg')
            
            # Use ECG intervals if available, else PPG
            intervals = ecg_intervals if len(ecg_intervals) > 0 else ppg_intervals
            if len(intervals) == 0:
                print(f"No valid intervals for {record_name}")
                continue
                
            # Prepare sequences
            X, y = prepare_sequences(intervals, seq_length)
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
                
        except Exception as e:
            print(f"Error loading {record_name}: {e}")
            continue
    
    if not X_all:
        raise Exception("No valid data loaded from any record")
    
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return X, y

# 2. Hybrid LSTM-CNN-Transformer Model
class HybridLSTMCNNTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_transformer_layers=2, 
                 lstm_hidden=32, num_cnn_filters=16, dropout=0.3):
        super(HybridLSTMCNNTransformer, self).__init__()
        self.d_model = d_model
        
        # CNN for local feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, num_cnn_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_cnn_filters),
            nn.MaxPool1d(2)
        )
        
        # LSTM
        self.lstm = nn.LSTM(num_cnn_filters, lstm_hidden, num_layers=1, batch_first=True)
        
        # Project to transformer dimension
        self.input_projection = nn.Linear(lstm_hidden, d_model)
        
        # Transformer
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_transformer_layers)
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, 1]
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # [batch, seq_len/2, num_filters]
        x, _ = self.lstm(x)  # [batch, seq_len/2, lstm_hidden]
        x = self.input_projection(x) * np.sqrt(self.d_model)  # [batch, seq_len/2, d_model]
        x = self.transformer(x)  # [batch, seq_len/2, d_model]
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x).squeeze()

# 3. Training Loop
def train_model(model, train_loader, val_loader, device, num_epochs=50, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    os.makedirs('bb_model_checkpoints', exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_mae += torch.abs(y_pred - y_batch).sum().item()
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.4f} sec')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'bb_model_checkpoints/best_bb_hybrid_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        scheduler.step()
        
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    return best_val_loss

# 4. Evaluation and Visualization
def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            y_true.extend(y_batch.numpy())
            y_pred.extend(pred.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    
    print(f'\n=== Test Results ===')
    print(f'Test MAE: {mae:.4f} seconds')
    print(f'Test MSE: {mse:.4f} seconds²')
    print(f'Average Actual Interval: {np.mean(y_true):.4f} seconds')
    print(f'Average Predicted Interval: {np.mean(y_pred):.4f} seconds')
    print(f'=== End of Test Results ===\n')
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(y_true[:100], label='Actual Interval (sec)', color='blue')
    plt.plot(y_pred[:100], label='Predicted Interval (sec)', color='orange', linestyle='--')
    plt.title('Actual vs Predicted Beat-to-Beat Intervals (First 100)')
    plt.xlabel('Sequence Index')
    plt.ylabel('Interval (sec)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    errors = y_pred - y_true
    plt.hist(errors, bins=50, color='purple', alpha=0.7)
    plt.title(f'Error Distribution (MAE={mae:.4f} sec)')
    plt.xlabel('Error (sec)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return mae, mse

# 5. Memory Management
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Main Function
def main():
    print(f'Initial memory usage: {get_memory_usage():.1f} MB')
    
    # Load and preprocess data
    seq_length = 10
    X, y = load_and_preprocess_data(seq_length=seq_length)
    
    print(f'Memory after data loading: {get_memory_usage():.1f} MB')
    cleanup_memory()
    
    # Data stats
    print(f'Data shapes: X={X.shape}, y={y.shape}')
    if X.shape[0] == 0:
        print("Error: No valid data extracted!")
        return
    print(f'X range: [{X.min():.3f}, {X.max():.3f}] seconds')
    print(f'y range: [{y.min():.3f}, {y.max():.3f}] seconds')
    
    # Split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # [batch, seq_len, 1]
    X_val = torch.FloatTensor(X_val).unsqueeze(-1)
    X_test = torch.FloatTensor(X_test).unsqueeze(-1)
    y_train = torch.FloatTensor(y_train)
    y_val = torch.FloatTensor(y_val)
    y_test = torch.FloatTensor(y_test)
    
    # Create DataLoaders
    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f'Memory after DataLoader creation: {get_memory_usage():.1f} MB')
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridLSTMCNNTransformer(
        input_size=1,
        d_model=64,
        nhead=4,
        num_transformer_layers=2,
        lstm_hidden=32,
        num_cnn_filters=16,
        dropout=0.3
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Memory after model creation: {get_memory_usage():.1f} MB')
    
    # Train model
    print('Starting training...')
    start_time = time.time()
    train_model(model, train_loader, val_loader, device, num_epochs=50, patience=10)
    print(f'Training completed in {time.time() - start_time:.2f} seconds')
    
    # Load best model
    print('Loading best model...')
    model.load_state_dict(torch.load('bb_model_checkpoints/best_bb_hybrid_model.pth'))
    
    # Evaluate
    print('\nEvaluating on test set...')
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()