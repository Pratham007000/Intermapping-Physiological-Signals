import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy import signal
from scipy.signal import welch
import pywt

# ---------- CONFIGURATION ----------
MAX_SAMPLES = 30000      # Number of samples to analyze
FS = 500                 # Sampling frequency (Hz)
CHECKPOINT_DIR = "checkpoints_cpu_high_accuracy"  # Directory for saving outputs

# Create checkpoint directory
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

# ---------- DATA ACQUISITION ----------
def download_cebs_database():
    db_dir = "cebsdb"
    record_name = "b001"
    record_path = os.path.join(db_dir, record_name + ".dat")
    
    try:
        if not os.path.exists(db_dir):
            print(f"Creating directory: {db_dir}")
            os.makedirs(db_dir, exist_ok=True)
        
        if not os.path.exists(record_path):
            print(f"Downloading CEBS database record: {record_name}")
            wfdb.dl_database('cebsdb', db_dir, records=[record_name])
            print("Download complete!")
        else:
            print(f"CEBS database record '{record_name}' already exists.")
        return record_name
    except Exception as e:
        print(f"Error downloading database: {str(e)}")
        return None

record_name = download_cebs_database()
if record_name is None:
    print("Failed to access the CEBS database. Exiting.")
    exit(1)

try:
    record = wfdb.rdrecord(record_name, pn_dir="cebsdb")
    print("Successfully loaded the record.")
except Exception as e:
    print(f"Error reading record: {str(e)}")
    exit(1)

print("Available signals:", record.sig_name)

required_signals = ['I', 'SCG']
for sig in required_signals:
    if sig not in record.sig_name:
        print(f"Required signal '{sig}' not found.")
        exit(1)

ecg = record.p_signal[:, record.sig_name.index('I')].astype(np.float64)  # Use float64 to avoid overflow
scg = record.p_signal[:, record.sig_name.index('SCG')].astype(np.float64)

# Limit samples
ecg = ecg[:MAX_SAMPLES]
scg = scg[:MAX_SAMPLES]
print(f"Using {len(ecg)} samples.")

# Check for NaN in raw data
if np.isnan(ecg).any() or np.isnan(scg).any():
    print("Warning: NaN detected in raw ECG or SCG data")
    ecg = np.nan_to_num(ecg, nan=0.0)
    scg = np.nan_to_num(scg, nan=0.0)

# ---------- PREPROCESSING FUNCTIONS ----------
def wavelet_denoise(signal_data, wavelet='db8', level=3, threshold_type='soft'):
    """Adaptive wavelet denoising"""
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(signal_data))) / np.log(level + 2)
    coeffs[1:] = [pywt.threshold(c, threshold, mode=threshold_type) for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs, wavelet)
    if np.isnan(denoised).any() or np.isinf(denoised).any():
        print("Warning: NaN or Inf in wavelet denoising")
        denoised = np.nan_to_num(denoised, nan=0.0)
    return denoised

def filter_signal(signal_data, fs=FS, lowcut=0.5, highcut=40.0, order=6):
    """Apply bandpass filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, signal_data)
    if np.isnan(filtered).any() or np.isinf(filtered).any():
        print("Warning: NaN or Inf in bandpass filter")
        filtered = np.nan_to_num(filtered, nan=0.0)
    return filtered

def robust_normalize(signal, percentile_clip=99.0):
    """Robust normalization with better handling of extreme values"""
    # Ensure float64 precision
    signal = signal.astype(np.float64)
    
    # Check for extreme values
    if np.any(np.abs(signal) > 1e6):
        print("Warning: Extreme values detected in signal before normalization")
        signal = np.clip(signal, -1e6, 1e6)
    
    # Clip outliers based on percentile
    threshold = np.percentile(np.abs(signal), percentile_clip)
    signal = np.clip(signal, -threshold, threshold)
    
    # Compute mean and std with clipping to avoid overflow
    mean = np.mean(signal)
    signal_centered = signal - mean
    # Clip centered signal to avoid large squares in std computation
    signal_centered = np.clip(signal_centered, -1e6, 1e6)
    std = np.std(signal_centered)
    
    if std == 0:
        print("Warning: Standard deviation is zero in normalization")
        # Instead of setting std=1.0, return the centered signal to preserve some variability
        normalized = signal_centered
    else:
        normalized = signal_centered / (std + 1e-10)
    
    if np.isnan(normalized).any() or np.isinf(normalized).any():
        print("Warning: NaN or Inf in normalization")
        normalized = np.nan_to_num(normalized, nan=0.0)
    
    return normalized

# Apply preprocessing with debugging for SCG
print("Applying preprocessing...")

# ECG preprocessing
ecg_denoised = wavelet_denoise(ecg, wavelet='db8', level=3)
ecg_filtered = filter_signal(ecg_denoised, fs=FS, lowcut=0.5, highcut=50.0)
ecg_processed = robust_normalize(ecg_filtered)

# SCG preprocessing with step-by-step debugging
print("\n--- SCG Preprocessing Debug ---")
print(f"Raw SCG stats: min={np.min(scg):.4f}, max={np.max(scg):.4f}, mean={np.mean(scg):.4f}, std={np.std(scg):.4f}")

scg_denoised = wavelet_denoise(scg, wavelet='db8', level=3)
print(f"After wavelet denoising: min={np.min(scg_denoised):.4f}, max={np.max(scg_denoised):.4f}, mean={np.mean(scg_denoised):.4f}, std={np.std(scg_denoised):.4f}")

scg_filtered = filter_signal(scg_denoised, fs=FS, lowcut=0.5, highcut=40.0)
print(f"After bandpass filter: min={np.min(scg_filtered):.4f}, max={np.max(scg_filtered):.4f}, mean={np.mean(scg_filtered):.4f}, std={np.std(scg_filtered):.4f}")

scg_processed = robust_normalize(scg_filtered)
print(f"After normalization: min={np.min(scg_processed):.4f}, max={np.max(scg_processed):.4f}, mean={np.mean(scg_processed):.4f}, std={np.std(scg_processed):.4f}")

# ---------- DIAGNOSTIC ANALYSIS ----------
# 1. Statistical Analysis
print("\n--- Statistical Analysis ---")
print(f"Raw ECG stats: min={np.min(ecg):.4f}, max={np.max(ecg):.4f}, mean={np.mean(ecg):.4f}, std={np.std(ecg):.4f}")
print(f"Processed ECG stats: min={np.min(ecg_processed):.4f}, max={np.max(ecg_processed):.4f}, mean={np.mean(ecg_processed):.4f}, std={np.std(ecg_processed):.4f}")
print(f"Raw SCG stats: min={np.min(scg):.4f}, max={np.max(scg):.4f}, mean={np.mean(scg):.4f}, std={np.std(scg):.4f}")
print(f"Processed SCG stats: min={np.min(scg_processed):.4f}, max={np.max(scg_processed):.4f}, mean={np.mean(scg_processed):.4f}, std={np.std(scg_processed):.4f}")

# 2. Time-Domain Visualization
# Plot raw and processed ECG
plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.plot(ecg[:2000], label='Raw ECG', color='blue')
plt.title("Raw ECG (First 2000 Samples)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(ecg_processed[:2000], label='Processed ECG', color='green')
plt.title("Processed ECG (First 2000 Samples)")
plt.xlabel("Sample")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, "ecg_data_inspection.png"))
plt.close()
print("ECG inspection plot saved as 'ecg_data_inspection.png'")

# Plot raw and processed SCG
plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.plot(scg[:2000], label='Raw SCG', color='blue')
plt.title("Raw SCG (First 2000 Samples)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(scg_processed[:2000], label='Processed SCG', color='green')
plt.title("Processed SCG (First 2000 Samples)")
plt.xlabel("Sample")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, "scg_data_inspection.png"))
plt.close()
print("SCG inspection plot saved as 'scg_data_inspection.png'")

# 3. Frequency-Domain Analysis (Power Spectral Density)
# ECG PSD
freqs_ecg, psd_ecg = welch(ecg, fs=FS, nperseg=1024)
freqs_ecg_proc, psd_ecg_proc = welch(ecg_processed, fs=FS, nperseg=1024)

plt.figure(figsize=(10, 6))
plt.semilogy(freqs_ecg, psd_ecg, label='Raw ECG', color='blue')
plt.semilogy(freqs_ecg_proc, psd_ecg_proc, label='Processed ECG', color='green')
plt.title("Power Spectral Density of ECG")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0, 60)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(CHECKPOINT_DIR, "ecg_psd.png"))
plt.close()
print("ECG PSD plot saved as 'ecg_psd.png'")

# SCG PSD
freqs_scg, psd_scg = welch(scg, fs=FS, nperseg=1024)
freqs_scg_proc, psd_scg_proc = welch(scg_processed, fs=FS, nperseg=1024)

plt.figure(figsize=(10, 6))
plt.semilogy(freqs_scg, psd_scg, label='Raw SCG', color='blue')
plt.semilogy(freqs_scg_proc, psd_scg_proc, label='Processed SCG', color='green')
plt.title("Power Spectral Density of SCG")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0, 60)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(CHECKPOINT_DIR, "scg_psd.png"))
plt.close()
print("SCG PSD plot saved as 'scg_psd.png'")

# 4. Temporal Alignment Between ECG and SCG
plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.plot(ecg[:2000], label='Raw ECG', color='blue')
plt.title("Raw ECG (First 2000 Samples)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(scg[:2000], label='Raw SCG', color='purple')
plt.title("Raw SCG (First 2000 Samples)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, "ecg_scg_alignment.png"))
plt.close()
print("ECG-SCG alignment plot saved as 'ecg_scg_alignment.png'")

# 5. Estimate Heart Rate from ECG
from scipy.signal import find_peaks
peaks, _ = find_peaks(ecg, distance=FS*0.6)  # Minimum distance of 0.6s between peaks (100 bpm max)
rr_intervals = np.diff(peaks) / FS  # RR intervals in seconds
heart_rate = 60 / np.mean(rr_intervals)  # Heart rate in bpm
print(f"\nEstimated Heart Rate from ECG: {heart_rate:.2f} bpm")
print(f"Average RR interval: {np.mean(rr_intervals):.3f} seconds ({np.mean(rr_intervals)*FS:.0f} samples)")

# Save summary to a text file
with open(os.path.join(CHECKPOINT_DIR, "data_diagnostic_summary.txt"), 'w') as f:
    f.write("ECG and SCG Data Diagnostic Summary\n")
    f.write("===================================\n\n")
    f.write("Statistical Analysis\n")
    f.write(f"Raw ECG stats: min={np.min(ecg):.4f}, max={np.max(ecg):.4f}, mean={np.mean(ecg):.4f}, std={np.std(ecg):.4f}\n")
    f.write(f"Processed ECG stats: min={np.min(ecg_processed):.4f}, max={np.max(ecg_processed):.4f}, mean={np.mean(ecg_processed):.4f}, std={np.std(ecg_processed):.4f}\n")
    f.write(f"Raw SCG stats: min={np.min(scg):.4f}, max={np.max(scg):.4f}, mean={np.mean(scg):.4f}, std={np.std(scg):.4f}\n")
    f.write(f"Processed SCG stats: min={np.min(scg_processed):.4f}, max={np.max(scg_processed):.4f}, mean={np.mean(scg_processed):.4f}, std={np.std(scg_processed):.4f}\n")
    f.write(f"\nEstimated Heart Rate from ECG: {heart_rate:.2f} bpm\n")
    f.write(f"Average RR interval: {np.mean(rr_intervals):.3f} seconds ({np.mean(rr_intervals)*FS:.0f} samples)\n")
print("Diagnostic summary saved to 'data_diagnostic_summary.txt'")