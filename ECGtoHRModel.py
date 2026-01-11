import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load ECG data from CSV file
try:
    ecg_data = pd.read_csv("ecg_data_20250701_172937.csv")["ECG Amplitude"].values
    ecg_data = ecg_data.astype(np.float32)
    if len(ecg_data) < 500:
        raise ValueError("ECG data length is less than 500 samples.")
except FileNotFoundError:
    print("Error: ecg_data_20250701_172937.csv not found. Please ensure the file is in the current directory.")
    exit()
except KeyError:
    print("Error: CSV file must contain an 'ECG Amplitude' column. Check the file structure.")
    exit()
except ValueError as e:
    print(f"Error: {e}")
    exit()

# Preprocess ECG data
def preprocess_ecg(ecg_data, window_size=500, epsilon=1e-8):
    ecg_normalized = (ecg_data - np.mean(ecg_data)) / (np.std(ecg_data) + epsilon)
    return ecg_normalized[:window_size]

# Define enhanced LSTM model for heart rate prediction
class ECGtoHRModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=512, output_size=1, num_layers=2, dropout_rate=0.5):
        super(ECGtoHRModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use last timestep output for sequence-to-one prediction
        return out

# Training function with sliding window, learning rate scheduling, and early stopping
def train_model(model, X_train, y_train, X_val, y_val, epochs=800, batch_size=4, lr=0.0002, patience=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    actual_batch_size = min(batch_size, len(X_train))
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        for i in range(0, len(X_train), actual_batch_size):
            batch_X = X_train[i:i + actual_batch_size]
            batch_y = y_train[i:i + actual_batch_size]
            
            if len(batch_X) == 0:
                continue
                
            batch_X_tensor = torch.tensor(batch_X, dtype=torch.float32).unsqueeze(-1)
            batch_y_tensor = torch.tensor(batch_y, dtype=torch.float32).unsqueeze(-1)
            
            optimizer.zero_grad()
            outputs = model(batch_X_tensor)
            loss = criterion(outputs, batch_y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val), actual_batch_size):
                batch_X = X_val[i:i + actual_batch_size]
                batch_y = y_val[i:i + actual_batch_size]
                
                if len(batch_X) == 0:
                    continue
                
                batch_X_tensor = torch.tensor(batch_X, dtype=torch.float32).unsqueeze(-1)
                batch_y_tensor = torch.tensor(batch_y, dtype=torch.float32).unsqueeze(-1)
                
                outputs = model(batch_X_tensor)
                loss = criterion(outputs, batch_y_tensor)
                total_val_loss += loss.item()
                num_val_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else float('inf')
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        if num_train_batches > 0 and (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        elif num_train_batches == 0:
            print(f"Warning: No training batches processed in epoch {epoch + 1}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            logging.info(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    return train_losses, val_losses, best_val_loss, epoch + 1

# Generate synthetic heart rate target (used only for training, not displayed)
def generate_synthetic_hr(ecg_data, fs=250):
    peaks, _ = find_peaks(ecg_data, height=0.25, distance=int(fs * 0.4))
    if len(peaks) < 2:
        logging.warning("Insufficient peaks detected, defaulting to 60 bpm")
        return np.full_like(ecg_data, 60.0)
    
    rr_intervals = np.diff(peaks) / fs
    hr = 60 / rr_intervals
    hr_smooth = gaussian_filter1d(hr, sigma=2)
    hr_extended = np.interp(np.arange(len(ecg_data)), np.linspace(0, len(ecg_data)-1, len(peaks)-1), hr_smooth)
    hr_extended = np.clip(hr_extended, 40, 180)
    return hr_extended

# Post-process predicted heart rate with smoothing
def smooth_hr(hr_data, sigma=2.5):
    return gaussian_filter1d(hr_data, sigma=sigma)

# Plotting function with metrics
def plot_ecg_predicted_hr(ecg_data, predicted_hr, train_losses, val_losses, best_val_loss, final_epoch, window_size=500, sampling_rate=250):
    time = np.arange(window_size) / sampling_rate
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # ECG and Predicted HR Plot
    ax1.plot(time, ecg_data, label='Actual ECG', color='blue', linewidth=1.5)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('ECG Amplitude (Normalized)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(labelsize=10)
    
    ax1.twinx()
    ax1.plot(time, predicted_hr, label='Predicted HR', color='green', linestyle='--', linewidth=1.5)
    ax1.set_ylabel('Predicted Value (Normalized)', fontsize=12, color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.tick_params(labelsize=10)
    
    ax1.set_title('ECG and Predicted Values (First 500 Timepoints)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    
    # Loss Plot
    epochs = range(1, len(train_losses) + 1)
    ax2.plot(epochs, train_losses, label='Training Loss', color='blue')
    ax2.plot(epochs, val_losses, label='Validation Loss', color='red')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(labelsize=10)
    ax2.legend(fontsize=10)
    ax2.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    
    # Add metrics text box
    textstr = (f'Best Validation Loss: {best_val_loss:.4f}\n'
               f'Final Epoch: {final_epoch}\n'
               f'Final Training Loss: {train_losses[-1]:.4f}\n'
               f'Final Validation Loss: {val_losses[-1]:.4f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()

# Main function
def main():
    window_size = 500
    ecg_data_subset = ecg_data[:window_size]
    
    # Preprocess ECG
    X_ecg = preprocess_ecg(ecg_data_subset)
    
    # Generate synthetic heart rate target with R-peak guidance
    y_target = generate_synthetic_hr(ecg_data_subset)
    
    # Create sliding window data with larger windows
    sliding_window_size = 300
    step_size = 15
    X, y = [], []
    for i in range(0, len(X_ecg) - sliding_window_size + 1, step_size):
        X.append(X_ecg[i:i + sliding_window_size])
        y.append(y_target[i:i + sliding_window_size].mean())  # Average HR over the window
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into training and validation sets (first 400 points for training, last 100 for validation)
    train_split = 400
    X_train = X[:train_split // step_size]
    y_train = y[:train_split // step_size]
    X_val = X[train_split // step_size:]
    y_val = y[train_split // step_size:]
    
    logging.info(f"Created {len(X_train)} training windows and {len(X_val)} validation windows of size {sliding_window_size}")
    logging.info(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    logging.info(f"Validation data shape: X_val={X_val.shape}, y_val={y_val.shape}")
    
    # Initialize model
    model = ECGtoHRModel(hidden_size=512, num_layers=2, dropout_rate=0.5)
    
    # Train model
    train_losses, val_losses, best_val_loss, final_epoch = train_model(model, X_train, y_train, X_val, y_val, epochs=800, lr=0.0002, patience=50)
    
    # Predict heart rate using sliding windows
    predicted_hr_values = []
    for i in range(0, len(X_ecg) - sliding_window_size + 1, step_size):
        window = X_ecg[i:i + sliding_window_size]
        sample = torch.tensor(window.reshape(1, sliding_window_size, 1), dtype=torch.float32)
        with torch.no_grad():
            hr_pred = model(sample).item()  # Single value per window
            predicted_hr_values.append(hr_pred)
    
    # Interpolate to match ECG length
    predicted_hr = np.interp(np.arange(len(X_ecg)), 
                           np.arange(0, len(X_ecg) - sliding_window_size + 1, step_size), 
                           predicted_hr_values)
    
    # Ensure predicted values are within a normalized range (for visualization purposes)
    predicted_hr = np.clip(predicted_hr, 0, 1)  # Normalized to avoid specific HR units
    
    # Smooth the predicted values
    predicted_hr = smooth_hr(predicted_hr)
    
    # Plot ECG and predicted values with metrics
    plot_ecg_predicted_hr(ecg_data_subset, predicted_hr, train_losses, val_losses, best_val_loss, final_epoch, window_size)

if __name__ == "__main__":
    main()