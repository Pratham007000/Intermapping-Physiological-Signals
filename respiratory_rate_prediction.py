import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.metrics import mean_absolute_error
import os
import time
from datetime import datetime

# Define device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LSTM model for direct RR prediction
class RR_LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout_rate=0.3):
        super(RR_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Load ECG and PPG data
ecg = np.loadtxt("bidmc01_ecg.csv", delimiter=",")
ppg = np.loadtxt("bidmc01_ppg.csv", delimiter=",")
assert ecg.shape == ppg.shape, "ECG and PPG must have equal length"

# Clean data: Remove NaN and inf values
ecg = np.nan_to_num(ecg, nan=np.nanmean(ecg), posinf=np.nanmax(ecg), neginf=np.nanmin(ecg))
ppg = np.nan_to_num(ppg, nan=np.nanmean(ppg), posinf=np.nanmax(ppg), neginf=np.nanmin(ppg))

# Clip extreme values
ecg = np.clip(ecg, -100, 100)
ppg = np.clip(ppg, -100, 100)

# Normalize signals
ecg = (ecg - ecg.mean()) / ecg.std()
ppg = (ppg - ppg.mean()) / ppg.std()

# Compute ground-truth RR using a sliding window FFT
fs = 125  # Sampling frequency (Hz)
window_len = 1250  # 10 seconds
step = 125  # 1 second steps
rr_signal = []

# Try FFT method first
try:
    for i in range(0, len(ppg) - window_len, step):
        ppg_window = ppg[i:i+window_len]
        # Ensure the window has no extreme values
        ppg_window = np.nan_to_num(ppg_window, nan=0.0, posinf=1.0, neginf=-1.0)
        N = len(ppg_window)
        yf = fft(ppg_window)
        if np.any(np.isnan(yf)) or np.any(np.isinf(yf)):
            rr = 15.0  # Default if FFT fails
        else:
            xf = fftfreq(N, 1/fs)[:N//2]
            power = np.abs(yf[:N//2])**2
            # Ensure power is valid
            power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)
            resp_band = (xf >= 0.1) & (xf <= 0.5)
            if not np.any(resp_band):
                rr = 15.0
            else:
                idx = np.argmax(power[resp_band])
                freq = xf[resp_band][idx]
                rr = freq * 60
                rr = np.clip(rr, 6, 30)
        rr_signal.append(rr)
except Exception as e:
    print(f"FFT method failed: {e}. Falling back to filtering method.")
    rr_signal = []
    # Fallback: Bandpass filter and peak detection
    try:
        lowcut, highcut = 0.1, 0.5
        b, a = butter(4, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
        resp_envelope = filtfilt(b, a, ppg)
        # Clean the envelope
        resp_envelope = np.nan_to_num(resp_envelope, nan=0.0, posinf=1.0, neginf=-1.0)
        # Compute RR using peak detection on 10-second windows
        for i in range(0, len(ppg) - window_len, step):
            envelope_window = resp_envelope[i:i+window_len]
            peaks, _ = find_peaks(envelope_window, distance=int(2.0 * fs))  # Min 2s between breaths
            if len(peaks) < 2:
                rr = 15.0
            else:
                intervals = np.diff(peaks) / fs
                rr = 60 / np.mean(intervals)
                rr = np.clip(rr, 6, 30)
            rr_signal.append(rr)
    except Exception as e:
        print(f"Filtering method also failed: {e}. Using default RR of 15 bpm.")
        rr_signal = [15.0] * ((len(ppg) - window_len) // step + 1)

# Interpolate RR signal
rr_signal = np.repeat(rr_signal, step)[:len(ppg)]

# Create sequences and assign RR values
def create_sequences_and_rr(ecg, ppg, rr_signal, seq_len=250):
    xs, rrs = [], []
    for i in range(0, len(ecg) - seq_len, seq_len // 2):
        xs.append(np.stack([ecg[i:i+seq_len], ppg[i:i+seq_len]], axis=1))
        seq_rr = np.mean(rr_signal[i:i+seq_len])
        if np.isnan(seq_rr) or np.isinf(seq_rr):
            seq_rr = 15.0
        rrs.append(seq_rr)
    return np.array(xs), np.array(rrs)

X, Y = create_sequences_and_rr(ecg, ppg, rr_signal, seq_len=250)
X = X.astype(np.float32)
Y = Y.astype(np.float32)

# Data validation and cleaning
print(f"Data shapes: X={X.shape}, Y={Y.shape}")
print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
print(f"Y range (RR in bpm): [{Y.min():.3f}, {Y.max():.3f}]")
print(f"Y mean (RR in bpm): {Y.mean():.3f}, Y std: {Y.std():.3f}")

# Check for NaN/inf values
if np.any(np.isnan(X)) or np.any(np.isinf(X)):
    print("Warning: NaN or inf values found in X")
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
    print("Warning: NaN or inf values found in Y")
    Y = np.nan_to_num(Y, nan=15.0, posinf=30.0, neginf=6.0)

# Additional clipping
X = np.clip(X, -10, 10)
Y = np.clip(Y, 6, 30)

# Train/validation/test split (70/15/15)
train_split = int(0.7 * len(X))
val_split = int(0.85 * len(X))

X_train, Y_train = X[:train_split], Y[:train_split]
X_val, Y_val = X[train_split:val_split], Y[train_split:val_split]
X_test, Y_test = X[val_split:], Y[val_split:]

print(f"Dataset sizes: Train={X_train.shape[0]}, Validation={X_val.shape[0]}, Test={X_test.shape[0]}")

# DataLoaders
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model, loss, optimizer
model_dir = "rr_model_checkpoints"
os.makedirs(model_dir, exist_ok=True)

model = RR_LSTM(input_size=2).to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Training parameters
num_epochs = 50
early_stopping_patience = 10
best_val_loss = float('inf')
early_stopping_counter = 0
best_model_path = os.path.join(model_dir, "best_rr_lstm_model.pth")

# Metrics storage
train_losses, val_losses, val_mae = [], [], []

# Training loop
print("Starting training...")
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch)
        y_batch = y_batch.view(-1, 1)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    total_val_loss = 0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            output = model(x_val)
            y_val = y_val.view(-1, 1)
            loss = criterion(output, y_val)
            total_val_loss += loss.item()
            val_preds.append(output.cpu().numpy())
            val_labels.append(y_val.cpu().numpy())
    
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # Calculate RR MAE
    val_preds = np.concatenate(val_preds).flatten()
    val_labels = np.concatenate(val_labels).flatten()
    val_mae.append(mean_absolute_error(val_labels, val_preds))
    
    # Save best model
    if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Model saved at epoch {epoch+1} with validation loss: {best_val_loss:.6f}")
    else:
        early_stopping_counter += 1
        if np.isnan(avg_val_loss):
            print(f"Warning: NaN validation loss at epoch {epoch+1}")
    
    scheduler.step(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val RR MAE: {val_mae[-1]:.2f} bpm")
    
    if early_stopping_counter >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

print(f"Training completed in {time.time() - start_time:.2f} seconds")

# Load best model
if os.path.exists(best_model_path):
    print(f"Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
else:
    print("Warning: No valid checkpoint found. Using current model state.")

# Test evaluation
print("\nEvaluating on test set...")
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        pred = model(x_test)
        test_preds.append(pred.cpu().numpy())
        test_labels.append(y_test.cpu().numpy())

test_preds = np.concatenate(test_preds).flatten()
test_labels = np.concatenate(test_labels).flatten()

test_mae = mean_absolute_error(test_labels, test_preds)
print(f"Test RR MAE: {test_mae:.2f} bpm")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"rr_results_{timestamp}.txt", 'w') as f:
    f.write(f"Test RR MAE: {test_mae:.2f} bpm\n")
    f.write(f"Model: RR_LSTM\n")
    f.write(f"Input: ECG + PPG (raw signals)\n")
    f.write(f"Sequence length: 250\n")

# Visualization
plt.figure(figsize=(15, 10))

# Plot 1: Predicted vs Actual RR for Test Set
plt.subplot(2, 2, 1)
plt.plot(test_labels[:100], label="Actual RR (bpm)", color='green')
plt.plot(test_preds[:100], label="Predicted RR (bpm)", color='red', linestyle='--')
plt.title("Actual vs Predicted RR (First 100 Sequences)")
plt.xlabel("Sequence Index")
plt.ylabel("RR (bpm)")
plt.ylim(0, 30)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Training and Validation Loss
plt.subplot(2, 2, 2)
epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MAE)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Validation RR MAE
plt.subplot(2, 2, 3)
plt.plot(epochs, val_mae, 'g-', label='RR MAE (bpm)')
plt.title('Validation RR MAE Over Time')
plt.xlabel('Epochs')
plt.ylabel('MAE (bpm)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: RR Scatter Plot
plt.subplot(2, 2, 4)
plt.scatter(test_labels, test_preds, alpha=0.5, s=10)
plt.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], 'r--')
plt.title(f'Predicted vs Actual RR (MAE={test_mae:.2f} bpm)')
plt.xlabel('Actual RR (bpm)')
plt.ylabel('Predicted RR (bpm)')
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"rr_results_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"Results saved to rr_results_{timestamp}.txt and rr_results_{timestamp}.png")