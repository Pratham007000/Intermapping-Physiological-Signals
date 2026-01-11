import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from lstm_ppg_model import ECGtoPPG_LSTM
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import os
import time
from datetime import datetime
import math
import pandas as pd

# Define device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load ECG data from original CSV file
def load_ecg_data():
    """Load ECG data from ecg_data_20250701_172937.csv"""
    try:
        data = pd.read_csv("ecg_data_20250701_172937.csv")
        ecg = data["ECG Amplitude"].values
        print(f"Loaded {len(ecg)} ECG samples from ecg_data_20250701_172937.csv")
        return ecg
    except FileNotFoundError:
        print("ecg_data_20250701_172937.csv not found. Please ensure the file is in the current directory.")
        exit()

# Generate synthetic PPG target from ECG for training
def generate_ppg_from_ecg(ecg_data, delay=100, pulse_width=80):
    """Generate synthetic PPG signal from ECG for training purposes"""
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    
    # Detect R-peaks in ECG
    peaks, _ = find_peaks(ecg_data, height=np.mean(ecg_data) + 0.5*np.std(ecg_data), distance=300)
    
    print(f"Detected {len(peaks)} R-peaks for PPG synthesis")
    
    # Generate PPG signal
    ppg_signal = np.zeros_like(ecg_data, dtype=np.float64)
    
    for peak in peaks:
        ppg_peak_time = peak + delay  # PPG delay after ECG R-peak
        
        if ppg_peak_time < len(ecg_data):
            # Create realistic PPG pulse
            rise_width = pulse_width // 4
            fall_width = pulse_width * 3 // 4
            
            # Rising edge
            rise_start = max(0, ppg_peak_time - rise_width//2)
            rise_end = min(len(ppg_signal), ppg_peak_time + rise_width//2)
            
            for i in range(rise_start, rise_end):
                rise_factor = (i - rise_start) / rise_width if rise_width > 0 else 0
                ppg_signal[i] += 0.8 * (1 - np.exp(-4 * rise_factor))
            
            # Falling edge with dicrotic notch
            fall_start = ppg_peak_time
            fall_end = min(len(ppg_signal), ppg_peak_time + fall_width)
            
            for i in range(fall_start, fall_end):
                fall_factor = (i - fall_start) / fall_width if fall_width > 0 else 0
                decay_value = 0.8 * np.exp(-2 * fall_factor)
                
                # Add dicrotic notch
                if 0.3 <= fall_factor <= 0.5:
                    notch_factor = 0.2 * np.sin(10 * np.pi * (fall_factor - 0.3))
                    decay_value += notch_factor * 0.3
                
                ppg_signal[i] += decay_value
    
    # Apply smoothing
    ppg_signal = gaussian_filter1d(ppg_signal, sigma=3.0)
    
    # Add respiratory modulation
    t = np.arange(len(ppg_signal), dtype=np.float64)
    resp_modulation = 0.05 * np.sin(2 * np.pi * t / 2000)  # Breathing effect
    ppg_signal = ppg_signal + resp_modulation
    
    # Normalize
    ppg_signal = (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)
    
    return ppg_signal

# Load ECG data
ecg = load_ecg_data()

# Generate synthetic PPG for training
print("Generating synthetic PPG target from ECG...")
ppg = generate_ppg_from_ecg(ecg)

# Normalize both signals
ecg = (ecg - ecg.mean()) / ecg.std()
ppg = (ppg - ppg.mean()) / ppg.std()

# Create sequences
def create_sequences(x, y, seq_len=100):
    xs, ys = [], []
    for i in range(len(x) - seq_len):
        xs.append(x[i:i+seq_len])
        ys.append(y[i:i+seq_len])
    return np.array(xs), np.array(ys)

X, Y = create_sequences(ecg, ppg, seq_len=100)
X = X[:, :, np.newaxis]
Y = Y[:, :, np.newaxis]

# Train/validation/test split (70/15/15)
train_split = int(0.7 * len(X))
val_split = int(0.85 * len(X))

X_train, Y_train = X[:train_split], Y[:train_split]
X_val, Y_val = X[train_split:val_split], Y[train_split:val_split]
X_test, Y_test = X[val_split:], Y[val_split:]

print(f"Dataset sizes: Train={X_train.shape[0]}, Validation={X_val.shape[0]}, Test={X_test.shape[0]}")

# DataLoaders
train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float())
val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(Y_val).float())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model, loss, optimizer
model_dir = "model_checkpoints"
os.makedirs(model_dir, exist_ok=True)

model = ECGtoPPG_LSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=5, factor=0.5
)

# Training parameters
num_epochs = 50
early_stopping_patience = 10

# Training tracking
train_losses = []
val_losses = []

# Early stopping variables
best_val_loss = float('inf')
early_stopping_counter = 0
best_model_path = os.path.join(model_dir, "best_lstm_model_original.pth")

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics between true and predicted values"""
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

# Training loop
print("Starting training...")
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    # Training phase
    model.train()
    total_train_loss = 0
    batch_count = 0
    
    for x_batch, y_batch in train_loader:
        batch_count += 1
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        output = model(x_batch)
        loss = criterion(output, y_batch)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        
        # Print progress every 20 batches
        if batch_count % 20 == 0:
            print(f"  Batch {batch_count}/{len(train_loader)} - Loss: {loss.item():.6f}", end='\r')
    
    # Calculate average training loss
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    total_val_loss = 0
    all_val_preds = []
    all_val_labels = []
    
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            
            val_output = model(x_val)
            val_loss = criterion(val_output, y_val)
            total_val_loss += val_loss.item()
            
            # Save predictions and labels for metrics calculation
            all_val_preds.append(val_output.cpu().numpy())
            all_val_labels.append(y_val.cpu().numpy())
    
    # Calculate average validation loss
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # Calculate validation metrics
    val_preds_array = np.concatenate(all_val_preds)
    val_labels_array = np.concatenate(all_val_labels)
    val_metrics = calculate_metrics(val_labels_array, val_preds_array)
    
    # Step the scheduler
    scheduler.step(avg_val_loss)
    
    # Check for early stopping and model saving
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
        
        # Save the best model
        torch.save(model.state_dict(), best_model_path)
        print(f"  Model saved at epoch {epoch+1} with validation loss: {best_val_loss:.6f}")
    else:
        early_stopping_counter += 1
        
    # Calculate epoch time
    epoch_time = time.time() - epoch_start
    
    # Print epoch summary
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
          f"Val RMSE: {val_metrics['rmse']:.4f}, Val R²: {val_metrics['r2']:.4f}, "
          f"Val Pearson: {val_metrics['pearson']:.4f}, "
          f"Time: {epoch_time:.2f}s")
    
    # Early stopping check
    if early_stopping_counter >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

# Training summary
total_time = time.time() - start_time
print(f"Training completed in {total_time:.2f} seconds")
print(f"Best validation loss: {best_val_loss:.6f}")

# Load the best model for prediction
try:
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print("Loaded best model successfully")
except:
    try:
        model.load_state_dict(torch.load(best_model_path, weights_only=False))
        print("Loaded best model using legacy method")
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Using current model state...")

# Generate predictions on test set
print("\nGenerating predictions...")
model.eval()

# Process test data
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

# Create visualization - ECG and Predicted PPG only
print("\nCreating ECG and Predicted PPG visualization...")

# Extract data for plotting (first 500 points)
actual_ecg = X_test[:100, :, 0].reshape(-1)[:500]  # ECG from test set
predicted_ppg = test_preds[:100, :, 0].reshape(-1)[:500]  # Predicted PPG

# Generate timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create the plot
plt.figure(figsize=(14, 8))

# Plot ECG and Predicted PPG
plt.plot(actual_ecg, label="ECG Signal", color='blue', linewidth=2, alpha=0.8)
plt.plot(predicted_ppg, label="Predicted PPG", color='red', linestyle='--', linewidth=2)

plt.title("ECG Signal and Predicted PPG (First 500 Timepoints)", fontsize=16, fontweight='bold')
plt.xlabel("Time", fontsize=14)
plt.ylabel("Normalized Amplitude", fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig(f"ecg_predicted_ppg_results_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"ECG and Predicted PPG plot saved to ecg_predicted_ppg_results_{timestamp}.png")
print("\nTraining and prediction complete!")
