import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from lstm_ppg_model import ECGtoPPG_LSTM
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import pandas as pd
from datetime import datetime
import math
import os

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load ECG data (reduced to 5000 samples for faster processing)
def load_ecg_data(max_samples=5000):
    try:
        data = pd.read_csv("ecg_data_20250701_172937.csv")
        ecg = data["ECG Amplitude"].values[-max_samples:]
        print(f"Loaded {len(ecg)} ECG samples from ecg_data_20250701_172937.csv")
        return ecg
    except FileNotFoundError:
        print("ecg_data_20250701_172937.csv not found. Please ensure the file is in the current directory.")
        exit()

# Optimized PPG generation with vectorized operations
def generate_ppg_from_ecg_fast(ecg_data, delay=100, pulse_width=80, cache_file="ppg_signal_5000.npy"):
    if os.path.exists(cache_file):
        print(f"Loading cached PPG signal from {cache_file}")
        ppg_signal = np.load(cache_file)
        if len(ppg_signal) == len(ecg_data):
            return ppg_signal
        else:
            print("Cached PPG signal length mismatch. Regenerating PPG...")
    
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    
    # Find peaks with relaxed parameters for faster processing
    peaks, _ = find_peaks(ecg_data, height=np.mean(ecg_data) + 0.3*np.std(ecg_data), distance=200)
    print(f"Detected {len(peaks)} R-peaks for PPG synthesis")
    
    # Vectorized PPG generation
    ppg_signal = np.zeros_like(ecg_data, dtype=np.float64)
    pulse_template = create_ppg_template(pulse_width)
    
    for peak in peaks:
        ppg_peak_time = peak + delay
        if ppg_peak_time < len(ecg_data):
            start_idx = max(0, ppg_peak_time - len(pulse_template)//2)
            end_idx = min(len(ppg_signal), start_idx + len(pulse_template))
            template_start = max(0, len(pulse_template)//2 - ppg_peak_time)
            template_end = template_start + (end_idx - start_idx)
            
            ppg_signal[start_idx:end_idx] += pulse_template[template_start:template_end]
    
    # Faster smoothing
    ppg_signal = gaussian_filter1d(ppg_signal, sigma=2.0)
    
    # Simplified respiratory modulation
    t = np.arange(len(ppg_signal), dtype=np.float64)
    resp_modulation = 0.05 * np.sin(2 * np.pi * t / 2000)
    ppg_signal += resp_modulation
    
    # Normalize
    ppg_signal = (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)
    
    # Save to cache
    np.save(cache_file, ppg_signal)
    print(f"Saved PPG signal to {cache_file}")
    return ppg_signal

# Create improved PPG pulse template for more realistic waveform
def create_ppg_template(pulse_width):
    """Create a physiologically realistic PPG pulse template"""
    # Adjust phase widths for better pulse morphology
    rise_width = int(pulse_width * 0.25)  # Sharp systolic upstroke (25%)
    systolic_plateau = int(pulse_width * 0.1)  # Brief plateau (10%)
    dicrotic_phase = int(pulse_width * 0.65)  # Extended diastolic phase (65%)
    
    total_width = rise_width + systolic_plateau + dicrotic_phase
    template = np.zeros(total_width)
    
    # 1. Systolic upstroke - Sharp rise to peak
    rise_indices = np.arange(rise_width)
    if rise_width > 0:
        # Use power function for sharper rise
        rise_factors = rise_indices / rise_width
        template[:rise_width] = np.power(rise_factors, 0.7)  # Sharper upstroke
    
    # 2. Brief systolic plateau
    plateau_start = rise_width
    plateau_end = rise_width + systolic_plateau
    template[plateau_start:plateau_end] = 1.0  # Peak amplitude
    
    # 3. Dicrotic phase with notch and diastolic peak
    dicrotic_start = plateau_end
    dicrotic_indices = np.arange(dicrotic_phase)
    if dicrotic_phase > 0:
        # Initial decay from systolic peak
        decay_factors = dicrotic_indices / dicrotic_phase
        
        # Base exponential decay
        base_decay = np.exp(-3 * decay_factors)
        
        # Add dicrotic notch at ~30% of dicrotic phase
        notch_position = 0.3
        notch_width = 0.15
        notch_mask = (decay_factors >= notch_position - notch_width/2) & (decay_factors <= notch_position + notch_width/2)
        
        # Create the notch (small dip)
        notch_depth = 0.15
        notch_phase = (decay_factors[notch_mask] - notch_position + notch_width/2) / notch_width * np.pi
        notch_modulation = -notch_depth * np.sin(notch_phase)
        
        # Add diastolic peak after the notch
        diastolic_peak_position = 0.5
        diastolic_peak_width = 0.2
        diastolic_mask = (decay_factors >= diastolic_peak_position - diastolic_peak_width/2) & (decay_factors <= diastolic_peak_position + diastolic_peak_width/2)
        
        # Create diastolic peak (secondary smaller peak)
        diastolic_amplitude = 0.25
        diastolic_phase = (decay_factors[diastolic_mask] - diastolic_peak_position + diastolic_peak_width/2) / diastolic_peak_width * np.pi
        diastolic_modulation = diastolic_amplitude * np.sin(diastolic_phase)
        
        # Combine all components
        dicrotic_waveform = base_decay.copy()
        if np.any(notch_mask):
            dicrotic_waveform[notch_mask] += notch_modulation
        if np.any(diastolic_mask):
            dicrotic_waveform[diastolic_mask] += diastolic_modulation
            
        template[dicrotic_start:dicrotic_start + dicrotic_phase] = dicrotic_waveform
    
    # Normalize template to prevent amplitude issues
    if np.max(template) > 0:
        template = template / np.max(template)
    
    # Apply slight smoothing to avoid sharp transitions
    from scipy.ndimage import gaussian_filter1d
    template = gaussian_filter1d(template, sigma=0.8)
    
    return template

# Optimized sequence creation (reduced sequences for faster processing)
def create_sequences_fast(x, y, seq_len=100, max_sequences=3000, cache_x="test_sequences_X_3000.npy", cache_y="test_sequences_Y_3000.npy"):
    if os.path.exists(cache_x) and os.path.exists(cache_y):
        print(f"Loading cached test sequences from {cache_x} and {cache_y}")
        X = np.load(cache_x)
        Y = np.load(cache_y)
        if X.shape[0] == max_sequences and Y.shape[0] == max_sequences:
            return X, Y
        else:
            print("Cached sequences length mismatch. Regenerating sequences...")
    
    # Vectorized sequence creation
    num_sequences = min(max_sequences, len(x) - seq_len)
    X = np.zeros((num_sequences, seq_len))
    Y = np.zeros((num_sequences, seq_len))
    
    for i in range(num_sequences):
        X[i] = x[i:i+seq_len]
        Y[i] = y[i:i+seq_len]
    
    # Save to cache
    np.save(cache_x, X)
    np.save(cache_y, Y)
    print(f"Saved test sequences to {cache_x} and {cache_y}")
    return X, Y

# Calculate metrics (unchanged)
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

if __name__ == '__main__':
    # Main execution
    print("Loading data...")
    ecg = load_ecg_data(max_samples=5000)  # Reduced from 10000
    print("Generating or loading cached PPG target...")
    ppg = generate_ppg_from_ecg_fast(ecg, cache_file="ppg_signal_5000.npy")
    
    # Normalize signals
    ecg = (ecg - ecg.mean()) / ecg.std()
    ppg = (ppg - ppg.mean()) / ppg.std()
    
    # Create sequences for test set (reduced from 9000 to 3000)
    print("Creating or loading cached test sequences...")
    X, Y = create_sequences_fast(ecg, ppg, seq_len=100, max_sequences=3000)
    X = X[:, :, np.newaxis]
    Y = Y[:, :, np.newaxis]
    print(f"Test dataset size: {X.shape[0]}")
    
    # Initialize model
    model = ECGtoPPG_LSTM().to(device)
    
    # Load the best model
    best_model_path = "model_checkpoints/best_lstm_model_original.pth"
    try:
        model.load_state_dict(torch.load(best_model_path, weights_only=True, map_location=device))
        print("Loaded best model successfully")
    except:
        try:
            model.load_state_dict(torch.load(best_model_path, weights_only=False, map_location=device))
            print("Loaded best model using legacy method")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Cannot proceed with testing. Please ensure the model file exists.")
            exit()
    
    # Generate predictions on test set (reduced num_workers to avoid multiprocessing issues)
    print("\nGenerating predictions...")
    model.eval()
    test_dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(Y).float())
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
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
    
    # Create visualization (reduced visualization size)
    print("\nCreating ECG and Target PPG visualization...")
    actual_ecg = X[:50, :, 0].reshape(-1)[:300]  # Reduced from 500 points
    target_ppg = Y[:50, :, 0].reshape(-1)[:300]   # Target PPG instead of predicted
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(12, 6))
    plt.plot(actual_ecg, label="ECG Signal", color='blue', linewidth=2, alpha=0.8)
    plt.plot(target_ppg, label="Target PPG", color='green', linewidth=2)
    plt.title("ECG Signal and Target PPG (Enhanced Morphology)", fontsize=14, fontweight='bold')
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Normalized Amplitude", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"ecg_target_ppg_results_{timestamp}.png", dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"ECG and Target PPG plot saved to ecg_target_ppg_results_{timestamp}.png")
    print("\nTesting complete!")
