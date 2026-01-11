import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import pandas as pd
import os
import time
import math
from datetime import datetime
from scipy.signal import butter, filtfilt, find_peaks

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Improved LSTM Model
class ImprovedECGtoPPG_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=512, num_layers=4, dropout=0.3):
        super(ImprovedECGtoPPG_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.layer_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Load ECG data
def load_ecg_data():
    try:
        data = pd.read_csv("ecg_data_20250701_172937.csv")
        ecg = data["ECG Amplitude"].values
        print(f"Loaded {len(ecg)} ECG samples from ecg_data_20250701_172937.csv")
        return ecg
    except FileNotFoundError:
        print("ecg_data_20250701_172937.csv not found. Please ensure the file is in the current directory.")
        exit()

# Improved PPG generation
def generate_improved_ppg_from_ecg(ecg_data, delay=110, pulse_width=85):
    b, a = butter(3, [0.5, 10], btype='band', fs=1000, output='ba')
    ecg_filtered = filtfilt(b, a, ecg_data)
    peaks, _ = find_peaks(ecg_filtered, height=np.mean(ecg_filtered) + 0.75*np.std(ecg_filtered), distance=290)
    print(f"Detected {len(peaks)} R-peaks for PPG synthesis")
    ppg_signal = np.zeros_like(ecg_data, dtype=np.float64)
    for peak in peaks:
        ppg_peak_time = peak + delay
        if ppg_peak_time < len(ecg_data):
            rise_width = pulse_width // 4
            fall_width = pulse_width * 3 // 4
            rise_start = max(0, ppg_peak_time - rise_width//2)
            rise_end = min(len(ppg_signal), ppg_peak_time + rise_width//2)
            for i in range(rise_start, rise_end):
                rise_factor = (i - rise_start) / rise_width if rise_width > 0 else 0
                ppg_signal[i] += 0.9 * (1 - np.exp(-5 * rise_factor))
            fall_start = ppg_peak_time
            fall_end = min(len(ppg_signal), ppg_peak_time + fall_width)
            for i in range(fall_start, fall_end):
                fall_factor = (i - fall_start) / fall_width if fall_width > 0 else 0
                decay_value = 0.9 * np.exp(-2.5 * fall_factor)
                if 0.3 <= fall_factor <= 0.5:
                    notch_factor = 0.25 * np.sin(11 * np.pi * (fall_factor - 0.3))
                    decay_value += notch_factor * 0.35
                ppg_signal[i] += decay_value
    ppg_signal = filtfilt(b, a, ppg_signal)
    t = np.arange(len(ppg_signal), dtype=np.float64)
    resp_modulation = 0.035 * np.sin(2 * np.pi * t / 1900)
    ppg_signal = ppg_signal + resp_modulation
    ppg_signal = (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)
    return ppg_signal

# Create sequences with overlap
def create_sequences(x, y, seq_len=150, overlap=0.7):
    step_size = int(seq_len * (1 - overlap))
    xs, ys = [], []
    for i in range(0, len(x) - seq_len + 1, step_size):
        xs.append(x[i:i+seq_len])
        ys.append(y[i:i+seq_len])
    return np.array(xs), np.array(ys)

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    pearson_corr, _ = pearsonr(y_true_flat, y_pred_flat)
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'pearson': pearson_corr}

# Load ECG and generate improved PPG
print("Loading data...")
ecg = load_ecg_data()
print("Generating improved synthetic PPG target from ECG...")
ppg = generate_improved_ppg_from_ecg(ecg)

# Normalize signals
ecg = (ecg - ecg.mean()) / ecg.std()
ppg = (ppg - ppg.mean()) / ppg.std()

# Create sequences
X, Y = create_sequences(ecg, ppg, seq_len=150, overlap=0.7)
X = X[:, :, np.newaxis]
Y = Y[:, :, np.newaxis]

# Train/validation/test split
train_split = int(0.7 * len(X))
val_split = int(0.85 * len(X))
X_train, X_val, X_test = X[:train_split], X[train_split:val_split], X[val_split:]
Y_train, Y_val, Y_test = Y[:train_split], Y[train_split:val_split], Y[val_split:]
print(f"Dataset sizes: Train={X_train.shape[0]}, Validation={X_val.shape[0]}, Test={X_test.shape[0]}")

# DataLoaders
train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float())
val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(Y_val).float())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, loss, optimizer
model_dir = "model_checkpoints"
os.makedirs(model_dir, exist_ok=True)
model = ImprovedECGtoPPG_LSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)  # Added L2 regularization
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=12, factor=0.5)

# Training parameters
num_epochs = 150
early_stopping_patience = 20
best_val_loss = float('inf')
early_stopping_counter = 0
best_model_path = os.path.join(model_dir, "best_lstm_model_improved.pth")

# Training loop
print("Starting training...")
start_time = time.time()
for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    total_train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Adjusted gradient clipping
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0
    all_val_preds, all_val_labels = [], []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            val_output = model(x_val)
            val_loss = criterion(val_output, y_val)
            total_val_loss += val_loss.item()
            all_val_preds.append(val_output.cpu().numpy())
            all_val_labels.append(y_val.cpu().numpy())
    avg_val_loss = total_val_loss / len(val_loader)
    val_preds = np.concatenate(all_val_preds)
    val_labels = np.concatenate(all_val_labels)
    val_metrics = calculate_metrics(val_labels, val_preds)

    scheduler.step(avg_val_loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"  Model saved at epoch {epoch+1} with validation loss: {best_val_loss:.6f}")
    else:
        early_stopping_counter += 1
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val RMSE: {val_metrics['rmse']:.4f}, Val R²: {val_metrics['r2']:.4f}, Val Pearson: {val_metrics['pearson']:.4f}, Time: {epoch_time:.2f}s")
    if early_stopping_counter >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

total_time = time.time() - start_time
print(f"Training completed in {total_time:.2f} seconds")
print(f"Best validation loss: {best_val_loss:.6f}")

# Test the model
model.load_state_dict(torch.load(best_model_path, weights_only=True))
print("Loaded best model successfully")

print("\nGenerating predictions...")
model.eval()
test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).float())
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

with torch.no_grad():
    all_test_preds, all_test_labels = [], []
    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        pred = model(x_test)
        all_test_preds.append(pred.cpu().numpy())
        all_test_labels.append(y_test.numpy())
    test_preds = np.concatenate(all_test_preds)
    test_labels = np.concatenate(all_test_labels)
    test_metrics = calculate_metrics(test_labels, test_preds)

    print(f"\nTest Set Performance:")
    print(f"Correlation: {test_metrics['pearson']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"R²: {test_metrics['r2']:.4f}")

# Visualization
print("\nCreating ECG and Predicted PPG visualization...")
actual_ecg = X_test[:100, :, 0].reshape(-1)[:500]
predicted_ppg = test_preds[:100, :, 0].reshape(-1)[:500]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.figure(figsize=(14, 8))
plt.plot(actual_ecg, label="ECG Signal", color='blue', linewidth=2, alpha=0.8)
plt.plot(predicted_ppg, label="Predicted PPG", color='red', linestyle='--', linewidth=2)
plt.title("ECG Signal and Predicted PPG (First 500 Timepoints)", fontsize=16, fontweight='bold')
plt.xlabel("Time", fontsize=14)
plt.ylabel("Normalized Amplitude", fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"ecg_predicted_ppg_results_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"ECG and Predicted PPG plot saved to ecg_predicted_ppg_results_{timestamp}.png")
print("\nTraining and prediction complete!")