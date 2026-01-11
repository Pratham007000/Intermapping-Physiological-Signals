import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from lstm_ppg_model import ECGtoPPG_LSTM
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import pandas as pd
from datetime import datetime
import math

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

# Generate synthetic PPG
def generate_ppg_from_ecg(ecg_data, delay=100, pulse_width=80):
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    peaks, _ = find_peaks(ecg_data, height=np.mean(ecg_data) + 0.5*np.std(ecg_data), distance=300)
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
                ppg_signal[i] += 0.8 * (1 - np.exp(-4 * rise_factor))
            fall_start = ppg_peak_time
            fall_end = min(len(ppg_signal), ppg_peak_time + fall_width)
            for i in range(fall_start, fall_end):
                fall_factor = (i - fall_start) / fall_width if fall_width > 0 else 0
                decay_value = 0.8 * np.exp(-2 * fall_factor)
                if 0.3 <= fall_factor <= 0.5:
                    notch_factor = 0.2 * np.sin(10 * np.pi * (fall_factor - 0.3))
                    decay_value += notch_factor * 0.3
                ppg_signal[i] += decay_value
    ppg_signal = gaussian_filter1d(ppg_signal, sigma=3.0)
    t = np.arange(len(ppg_signal), dtype=np.float64)
    resp_modulation = 0.05 * np.sin(2 * np.pi * t / 2000)
    ppg_signal = ppg_signal + resp_modulation
    ppg_signal = (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)
    return ppg_signal

# Create sequences
def create_sequences(x, y, seq_len=100):
    xs, ys = [], []
    for i in range(len(x) - seq_len):
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
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr
    }

# Load ECG and generate PPG
print("Loading data...")
ecg = load_ecg_data()
print("Generating synthetic PPG target from ECG...")
ppg = generate_ppg_from_ecg(ecg)

# Normalize signals
ecg = (ecg - ecg.mean()) / ecg.std()
ppg = (ppg - ppg.mean()) / ppg.std()

# Create sequences
X, Y = create_sequences(ecg, ppg, seq_len=100)
X = X[:, :, np.newaxis]
Y = Y[:, :, np.newaxis]

# Test split (use the last 15% of data as in original script)
test_split = int(0.85 * len(X))
X_test, Y_test = X[test_split:], Y[test_split:]
print(f"Test dataset size: {X_test.shape[0]}")

# Initialize model
model = ECGtoPPG_LSTM().to(device)

# Load the best model
best_model_path = "model_checkpoints/best_lstm_model_original.pth"
try:
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print("Loaded best model successfully")
except:
    try:
        model.load_state_dict(torch.load(best_model_path, weights_only=False))
        print("Loaded best model using legacy method")
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Cannot proceed with testing. Please ensure the model file exists.")
        exit()

# Generate predictions on test set
print("\nGenerating predictions...")
model.eval()
test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).float())
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

with torch.no_grad():
    all_test_preds = []
    all_test_labels = []
    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        pred = model(x_test)
        all_test_preds.append(pred.cpu().numpy())
        all_test_labels.append(y_test.numpy())
    
    # Concatenate predictions
    test_preds = np.concatenate(all_test_preds)
    test_labels = np.concatenate(all_test_labels)
    
    # Calculate test metrics
    test_metrics = calculate_metrics(test_labels, test_preds)
    
    print(f"\nTest Set Performance:")
    print(f"Correlation: {test_metrics['pearson']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"R²: {test_metrics['r2']:.4f}")

# Create visualization
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
print("\nTesting complete!")
