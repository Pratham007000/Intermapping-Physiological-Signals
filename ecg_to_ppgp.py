import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# Load ECG data from CSV file
try:
    ecg_data = pd.read_csv("ecg_data_20250701_172937.csv")["ECG Amplitude"].values
    ecg_data = ecg_data.astype(np.float32)
except FileNotFoundError:
    print("Error: ecg_data_20250701_172937.csv not found. Please ensure the file is in the current directory.")
    exit()
except KeyError:
    print("Error: CSV file must contain an 'ECG Amplitude' column. Check the file structure.")
    exit()

# Preprocess ECG data
def preprocess_ecg(ecg_data, window_size=500):
    ecg_normalized = (ecg_data - np.mean(ecg_data)) / np.std(ecg_data)
    return ecg_normalized[:window_size]

# Define enhanced LSTM model
class ECGtoPPGModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=2, dropout_rate=0.3):
        super(ECGtoPPGModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# Training function with sliding window
def train_model(model, X, y, epochs=200, batch_size=32, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Adjust batch size if needed
    actual_batch_size = min(batch_size, len(X))
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # Process all data in batches
        for i in range(0, len(X), actual_batch_size):
            batch_X = X[i:i + actual_batch_size]
            batch_y = y[i:i + actual_batch_size]
            
            if len(batch_X) == 0:
                continue
                
            batch_X_tensor = torch.tensor(batch_X, dtype=torch.float32).unsqueeze(-1)
            batch_y_tensor = torch.tensor(batch_y, dtype=torch.float32).unsqueeze(-1)
            
            optimizer.zero_grad()
            outputs = model(batch_X_tensor)
            loss = criterion(outputs, batch_y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Safe division to avoid zero division error
        if num_batches > 0 and (epoch + 1) % 20 == 0:
            avg_loss = total_loss / num_batches
            print(f'Epoch [{epoch + 1}/{epochs}], Avg Loss: {avg_loss:.4f}')
        elif num_batches == 0:
            print(f"Warning: No batches processed in epoch {epoch + 1}")

# Generate improved synthetic target
def generate_synthetic_target(ecg_data, delay=50, sigma=5, scale_factor=0.5):
    # Detect R-peaks to guide the target
    peaks, _ = find_peaks(ecg_data, height=0.5, distance=50)
    target = np.zeros_like(ecg_data)
    for peak in peaks:
        if peak + delay < len(ecg_data):
            target[peak + delay] = ecg_data[peak] * scale_factor  # Amplify R-peak influence
    target = gaussian_filter1d(target, sigma=sigma)
    target = np.roll(target, delay)
    target[:delay] = target[delay]  # Handle edge effects
    target = (target - np.mean(target)) / np.std(target)  # Normalize
    return target

# Post-process predicted PPG with smoothing
def smooth_ppg(ppg_data, sigma=2):
    return gaussian_filter1d(ppg_data, sigma=sigma)

# Plotting function
def plot_ecg_predicted_ppg(ecg_data, predicted_ppg, window_size=500, sampling_rate=250):
    time = np.arange(window_size) / sampling_rate
    plt.figure(figsize=(10, 6))
    plt.plot(time, ecg_data, label='Actual ECG', color='blue', linewidth=1.5)
    plt.plot(time, predicted_ppg, label='Predicted PPG', color='red', linestyle='--', linewidth=1.5)
    
    plt.title('ECG and Predicted PPG (First 500 Timepoints)', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude (Normalized)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

# Main function
def main():
    window_size = 500
    ecg_data_subset = ecg_data[:window_size]
    
    # Preprocess ECG
    X_ecg = preprocess_ecg(ecg_data_subset, window_size)
    
    # Generate synthetic target with R-peak guidance
    y_target = generate_synthetic_target(ecg_data_subset, delay=50, sigma=5, scale_factor=0.5)
    
    # Create sliding window data with smaller windows
    sliding_window_size = 100  # Smaller window size for sliding
    step_size = 25  # Step size for sliding window
    X, y = [], []
    
    # Create multiple windows by sliding through the data
    for i in range(0, len(X_ecg) - sliding_window_size + 1, step_size):
        X.append(X_ecg[i:i + sliding_window_size])
        y.append(y_target[i:i + sliding_window_size])
    
    # Convert to numpy arrays (all sublists now have the same length)
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} training windows of size {sliding_window_size}")
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    
    # Initialize model
    model = ECGtoPPGModel(hidden_size=64, num_layers=2, dropout_rate=0.3)
    
    # Train model
    train_model(model, X, y, epochs=200, lr=0.001)
    
    # Predict PPG for the first 500 samples
    sample = torch.tensor(X_ecg.reshape(1, window_size, 1), dtype=torch.float32)
    with torch.no_grad():
        predicted_ppg = model(sample).numpy().flatten()
    
    # Smooth the predicted PPG
    predicted_ppg = smooth_ppg(predicted_ppg, sigma=2)
    
    # Plot ECG and predicted PPG
    plot_ecg_predicted_ppg(X_ecg, predicted_ppg, window_size)

if __name__ == "__main__":
    main()