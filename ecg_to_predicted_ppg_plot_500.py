import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# Load ECG data from CSV file with fallback to realistic data generation
def load_or_generate_data():
    """Load ECG data or generate realistic synthetic data if file not found."""
    try:
        # Try to load existing data
        data = pd.read_csv("realistic_ecg_ppg_data.csv")
        ecg_data = data["ECG Amplitude"].values.astype(np.float32)
        ppg_data = data["PPG Amplitude"].values.astype(np.float32) if "PPG Amplitude" in data.columns else None
        print(f"✓ Loaded {len(ecg_data)} ECG samples from realistic_ecg_ppg_data.csv")
        if ppg_data is not None:
            print(f"✓ Also loaded actual PPG data for comparison")
        return ecg_data, ppg_data
    except FileNotFoundError:
        print("📁 No data file found, generating realistic synthetic ECG data...")
        # Generate realistic synthetic ECG data
        duration = 10  # seconds
        sampling_rate = 500  # Hz
        heart_rate = 75  # BPM
        
        # Time array
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        # Generate realistic ECG with R-peaks
        ecg_signal = np.zeros_like(t)
        rr_interval = 60 / heart_rate  # seconds between R-peaks
        
        # Add R-peaks at regular intervals with slight variability
        peak_times = np.arange(0.5, duration, rr_interval)
        peak_times += np.random.normal(0, 0.02, len(peak_times))  # Add HRV
        
        for peak_time in peak_times:
            if 0 <= peak_time < duration:
                peak_idx = int(peak_time * sampling_rate)
                
                # Add realistic PQRST complex
                # P wave
                p_start = peak_idx - int(0.12 * sampling_rate)
                if p_start >= 0:
                    for i in range(int(0.08 * sampling_rate)):
                        if p_start + i < len(ecg_signal):
                            ecg_signal[p_start + i] += 0.1 * np.exp(-(i - 20)**2 / 100)
                
                # QRS complex (main R-peak)
                qrs_width = int(0.08 * sampling_rate)
                qrs_start = max(0, peak_idx - qrs_width//2)
                qrs_end = min(len(ecg_signal), peak_idx + qrs_width//2)
                
                # Q wave (small negative)
                if qrs_start < len(ecg_signal):
                    ecg_signal[qrs_start] -= 0.2
                
                # R wave (large positive)
                if peak_idx < len(ecg_signal):
                    ecg_signal[peak_idx] += 1.0 + np.random.normal(0, 0.05)
                
                # S wave (negative)
                s_idx = peak_idx + int(0.02 * sampling_rate)
                if s_idx < len(ecg_signal):
                    ecg_signal[s_idx] -= 0.3
                
                # T wave
                t_start = peak_idx + int(0.15 * sampling_rate)
                for i in range(int(0.15 * sampling_rate)):
                    if t_start + i < len(ecg_signal):
                        ecg_signal[t_start + i] += 0.2 * np.exp(-(i - 30)**2 / 200)
        
        # Add realistic noise and baseline wander
        baseline_wander = 0.05 * np.sin(2 * np.pi * 0.5 * t)
        measurement_noise = np.random.normal(0, 0.03, len(t))
        ecg_signal += baseline_wander + measurement_noise
        
        # Apply bandpass filter
        nyquist = sampling_rate / 2
        low_cutoff = 0.5 / nyquist
        high_cutoff = 40 / nyquist
        b, a = butter(4, [low_cutoff, high_cutoff], btype='band')
        ecg_signal = filtfilt(b, a, ecg_signal)
        
        # Normalize
        ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
        ecg_signal = ecg_signal * 0.5 + 1.0  # Scale to reasonable amplitude
        
        print(f"✓ Generated {len(ecg_signal)} synthetic ECG samples ({duration}s at {sampling_rate}Hz)")
        print(f"💓 Heart rate: {heart_rate} BPM, R-peaks: {len(peak_times)}")
        
        return ecg_signal.astype(np.float32), None

ecg_data, actual_ppg_data = load_or_generate_data()

# Enhanced ECG preprocessing
def preprocess_ecg(ecg_data, window_size=500, apply_filter=True):
    """Enhanced ECG preprocessing with filtering and robust normalization."""
    
    # Select window
    ecg_segment = ecg_data[:window_size]
    
    if apply_filter:
        # Apply additional filtering if needed
        if len(ecg_segment) > 100:  # Only filter if we have enough samples
            # Remove any remaining high-frequency noise
            ecg_segment = gaussian_filter1d(ecg_segment, sigma=0.5)
    
    # Robust normalization using median and MAD (Median Absolute Deviation)
    median_val = np.median(ecg_segment)
    mad_val = np.median(np.abs(ecg_segment - median_val))
    
    if mad_val > 0:
        # Use MAD for robust normalization
        ecg_normalized = (ecg_segment - median_val) / (1.4826 * mad_val)
    else:
        # Fallback to standard normalization
        ecg_normalized = (ecg_segment - np.mean(ecg_segment)) / np.std(ecg_segment)
    
    return ecg_normalized

# Enhanced LSTM model with better architecture
class ECGtoPPGModel(nn.Module):
    """Improved LSTM model for ECG to PPG conversion."""
    
    def __init__(self, input_size=1, hidden_size=128, output_size=1, num_layers=2, dropout_rate=0.2):
        super(ECGtoPPGModel, self).__init__()
        
        # LSTM layers with residual connections
        self.lstm = nn.LSTM(
            input_size, hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False  # Causal system for real-time processing
        )
        
        # Output projection with intermediate layer
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size),
            nn.Tanh()  # Constrain output range
        )
        
        # Layer normalization for training stability
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Output projection
        output = self.fc_layers(lstm_out)
        
        return output

# Enhanced training function with validation
def train_model(model, X, y, epochs=150, batch_size=16, lr=0.001, val_split=0.2):
    """Enhanced training with validation and early stopping."""
    
    # Split data for validation
    split_idx = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"🎯 Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.7)
    criterion = nn.MSELoss()
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    actual_batch_size = min(batch_size, len(X_train))
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
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
            
            train_loss += loss.item()
            num_train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
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
                val_loss += loss.item()
                num_val_batches += 1
        
        # Calculate average losses
        avg_train_loss = train_loss / max(1, num_train_batches)
        avg_val_loss = val_loss / max(1, num_val_batches)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Progress reporting
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1:3d}/{epochs}] | '
                  f'Train: {avg_train_loss:.4f} | '
                  f'Val: {avg_val_loss:.4f} | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"⏹ Early stopping at epoch {epoch + 1}")
            break
    
    return train_losses, val_losses

# Generate realistic PPG target from ECG
def generate_synthetic_target(ecg_data, delay=100, sigma=3.0, scale_factor=0.8):
    """Generate physiologically realistic PPG target from ECG signal."""
    
    # Enhanced R-peak detection with multiple methods
    # Method 1: Standard peak detection
    peaks1, _ = find_peaks(ecg_data, height=np.mean(ecg_data) + 0.3*np.std(ecg_data), distance=60)
    
    # Method 2: Derivative-based detection
    diff_ecg = np.abs(np.diff(ecg_data))
    peaks2, _ = find_peaks(diff_ecg, height=np.mean(diff_ecg) + 2*np.std(diff_ecg), distance=60)
    
    # Combine and validate peaks
    all_peaks = np.concatenate([peaks1, peaks2])
    all_peaks = np.unique(all_peaks)
    
    # Remove peaks too close together
    if len(all_peaks) > 1:
        valid_peaks = [all_peaks[0]]
        for peak in all_peaks[1:]:
            if peak - valid_peaks[-1] >= 50:  # At least 50 samples apart
                valid_peaks.append(peak)
        peaks = np.array(valid_peaks)
    else:
        peaks = all_peaks
    
    print(f"🫀 Detected {len(peaks)} R-peaks for PPG synthesis")
    
    # Generate PPG target signal
    target = np.zeros_like(ecg_data)
    
    for peak in peaks:
        ppg_peak_time = peak + delay  # PPG delay after ECG R-peak
        
        if ppg_peak_time < len(ecg_data):
            # Create realistic PPG pulse morphology
            pulse_width = 80  # Width of PPG pulse
            
            # Systolic upstroke (sharp rise)
            rise_width = pulse_width // 4
            rise_start = max(0, ppg_peak_time - rise_width//2)
            rise_end = min(len(target), ppg_peak_time + rise_width//2)
            
            for i in range(rise_start, rise_end):
                # Exponential rise
                rise_factor = (i - rise_start) / rise_width
                target[i] += scale_factor * (1 - np.exp(-4 * rise_factor))
            
            # Diastolic decay (slower fall with dicrotic notch)
            fall_width = pulse_width * 3 // 4
            fall_start = ppg_peak_time
            fall_end = min(len(target), ppg_peak_time + fall_width)
            
            for i in range(fall_start, fall_end):
                # Exponential decay
                fall_factor = (i - fall_start) / fall_width
                decay_value = scale_factor * np.exp(-2 * fall_factor)
                
                # Add dicrotic notch (small secondary peak)
                if 0.3 <= fall_factor <= 0.5:
                    notch_factor = 0.2 * np.sin(10 * np.pi * (fall_factor - 0.3))
                    decay_value += notch_factor * scale_factor * 0.3
                
                target[i] += decay_value
    
    # Apply smoothing
    target = gaussian_filter1d(target, sigma=sigma)
    
    # Add realistic PPG characteristics
    # Respiratory modulation (breathing effect)
    t = np.arange(len(target))
    resp_modulation = 0.05 * np.sin(2 * np.pi * t / 1000)  # ~1 Hz breathing
    target += resp_modulation
    
    # Normalize target
    target = (target - np.mean(target)) / np.std(target)
    
    return target

# Post-process predicted PPG with smoothing
def smooth_ppg(ppg_data, sigma=2):
    return gaussian_filter1d(ppg_data, sigma=sigma)

# Enhanced plotting function with actual PPG comparison
def plot_ecg_predicted_ppg(ecg_data, predicted_ppg, actual_ppg=None, window_size=500, sampling_rate=500):
    """Plot ECG and predicted PPG, with optional actual PPG comparison."""
    
    time = np.arange(window_size) / sampling_rate
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top subplot: ECG signal
    ax1.plot(time, ecg_data, label='ECG Signal', color='blue', linewidth=1.5, alpha=0.8)
    ax1.set_title('ECG Signal (First 500 Timepoints)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('ECG Amplitude', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Bottom subplot: PPG comparison
    if actual_ppg is not None:
        ax2.plot(time, actual_ppg, label='Actual PPG', color='green', linewidth=2, alpha=0.8)
        ax2.plot(time, predicted_ppg, label='Predicted PPG', color='red', linestyle='--', linewidth=2, alpha=0.9)
        
        # Calculate correlation for comparison
        correlation = np.corrcoef(actual_ppg, predicted_ppg)[0, 1]
        rmse = np.sqrt(np.mean((actual_ppg - predicted_ppg)**2))
        
        ax2.set_title(f'Actual vs Predicted PPG (Correlation: {correlation:.3f}, RMSE: {rmse:.3f})', 
                     fontsize=14, fontweight='bold')
    else:
        ax2.plot(time, predicted_ppg, label='Predicted PPG', color='red', linewidth=2, alpha=0.9)
        ax2.set_title('Predicted PPG Signal', fontsize=14, fontweight='bold')
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('PPG Amplitude (Normalized)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print performance metrics if actual PPG is available
    if actual_ppg is not None:
        print(f"\n📊 Performance Metrics:")
        print(f"Correlation: {correlation:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {np.mean(np.abs(actual_ppg - predicted_ppg)):.4f}")

# Main function
def main():
    window_size = 500
    ecg_data_subset = ecg_data[:window_size]
    
    # Preprocess ECG
    X_ecg = preprocess_ecg(ecg_data_subset, window_size)
    
    # Use actual PPG if available, otherwise generate synthetic target
    if actual_ppg_data is not None:
        # Use actual PPG as target (normalize it)
        actual_ppg_subset = actual_ppg_data[:window_size]
        y_target = (actual_ppg_subset - np.mean(actual_ppg_subset)) / np.std(actual_ppg_subset)
        print("🎯 Using actual PPG data as training target")
    else:
        # Generate synthetic target with R-peak guidance
        y_target = generate_synthetic_target(ecg_data_subset, delay=50, sigma=5, scale_factor=0.5)
        print("🎯 Using synthetic PPG target")
    
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
    
    # Prepare actual PPG for comparison if available
    actual_ppg_for_plot = None
    if actual_ppg_data is not None:
        actual_ppg_subset = actual_ppg_data[:window_size]
        actual_ppg_for_plot = (actual_ppg_subset - np.mean(actual_ppg_subset)) / np.std(actual_ppg_subset)
    
    # Plot ECG and predicted PPG with comparison
    plot_ecg_predicted_ppg(X_ecg, predicted_ppg, actual_ppg_for_plot, window_size, sampling_rate=500)

if __name__ == "__main__":
    main()