import wfdb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from scipy import signal
import matplotlib.pyplot as plt
import os
import sys
import time
from tqdm import tqdm
import math
from pathlib import Path
import pywt
from improved_ecg_to_scg_model import ImprovedECGtoSCG, AdvancedLoss
from sklearn.model_selection import train_test_split
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import warnings
warnings.filterwarnings('ignore')

# ---------- ENHANCED CONFIGURATION ----------
MAX_SAMPLES = 50000         # Increased samples for better training
SEQ_LEN = 128               # Optimal sequence length for cardiac signals
STEP_SIZE = 32              # Overlapping sequences for data augmentation
BATCH_SIZE = 16             # Larger batch size for stable training
EPOCHS = 200                # More epochs for convergence
LEARNING_RATE = 0.001       # Optimal learning rate
WEIGHT_DECAY = 1e-5         # L2 regularization
EARLY_STOPPING_PATIENCE = 25
CHECKPOINT_DIR = "improved_checkpoints"
FS = 500                    # Sampling frequency

# Advanced training parameters
GRADIENT_CLIP = 1.0
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
AUGMENT_PROB = 0.7

# ---------- DEVICE SETUP ----------
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Create checkpoint directory
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

# ---------- ADVANCED DATA AUGMENTATION ----------
class AdvancedDataAugmentation:
    def __init__(self, fs=FS):
        self.fs = fs
        
    def add_noise(self, signal, noise_level=0.05):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level, signal.shape)
        return signal + noise
    
    def time_shift(self, signal, max_shift=10):
        """Random time shifting"""
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(signal, shift)
    
    def amplitude_scale(self, signal, scale_range=(0.8, 1.2)):
        """Random amplitude scaling"""
        scale = np.random.uniform(*scale_range)
        return signal * scale
    
    def frequency_warp(self, signal, warp_factor=0.1):
        """Frequency domain warping"""
        # Simple implementation using interpolation
        length = len(signal)
        indices = np.arange(length)
        warped_indices = indices + warp_factor * np.sin(2 * np.pi * indices / length)
        warped_indices = np.clip(warped_indices, 0, length - 1)
        return np.interp(indices, warped_indices, signal)
    
    def baseline_wander(self, signal, amplitude=0.1, frequency=0.5):
        """Add baseline wander"""
        t = np.arange(len(signal)) / self.fs
        baseline = amplitude * np.sin(2 * np.pi * frequency * t)
        return signal + baseline
    
    def high_freq_noise(self, signal, noise_level=0.02, freq_range=(50, 100)):
        """Add high-frequency noise"""
        t = np.arange(len(signal)) / self.fs
        freq = np.random.uniform(*freq_range)
        hf_noise = noise_level * np.sin(2 * np.pi * freq * t)
        return signal + hf_noise
    
    def apply_random_augmentation(self, signal, prob=0.7):
        """Apply random combination of augmentations"""
        if np.random.random() < prob:
            augmentations = [
                lambda x: self.add_noise(x, np.random.uniform(0.01, 0.08)),
                lambda x: self.time_shift(x, np.random.randint(5, 15)),
                lambda x: self.amplitude_scale(x, (0.85, 1.15)),
                lambda x: self.frequency_warp(x, np.random.uniform(-0.05, 0.05)),
                lambda x: self.baseline_wander(x, np.random.uniform(0.05, 0.15)),
                lambda x: self.high_freq_noise(x, np.random.uniform(0.01, 0.03))
            ]
            
            # Apply 1-3 random augmentations
            num_augs = np.random.randint(1, 4)
            selected_augs = np.random.choice(augmentations, num_augs, replace=False)
            
            for aug in selected_augs:
                signal = aug(signal)
                
        return signal

# ---------- ENHANCED DATA PREPROCESSING ----------
def advanced_signal_preprocessing(signal, fs=FS):
    """Advanced preprocessing pipeline"""
    # Remove outliers using robust statistics
    q75, q25 = np.percentile(signal, [75, 25])
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    signal = np.clip(signal, lower_bound, upper_bound)
    
    # Multi-level wavelet denoising
    coeffs = pywt.wavedec(signal, 'db8', level=6)
    # Soft thresholding for denoising
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    signal = pywt.waverec(coeffs, 'db8')
    
    # Advanced filtering
    # High-pass filter to remove baseline wander
    sos_hp = signal.butter(4, 0.5, 'high', fs=fs, output='sos')
    signal = signal.sosfilt(sos_hp, signal)
    
    # Band-pass filter for signal of interest
    if 'scg' in locals() or 'SCG' in str(type(signal)):
        # SCG specific filtering
        sos_bp = signal.butter(6, [0.5, 30], 'band', fs=fs, output='sos')
    else:
        # ECG specific filtering
        sos_bp = signal.butter(6, [0.5, 50], 'band', fs=fs, output='sos')
    
    signal = signal.sosfilt(sos_bp, signal)
    
    # Robust normalization
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    signal = (signal - median) / (1.4826 * mad + 1e-10)
    
    return signal

# ---------- DATA LOADING AND PREPROCESSING ----------
def download_and_load_data():
    """Download and load CEBS database"""
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
        
        record = wfdb.rdrecord(record_name, pn_dir="cebsdb")
        print("Successfully loaded the record.")
        print("Available signals:", record.sig_name)
        
        return record
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

record = download_and_load_data()
if record is None:
    print("Failed to load data. Exiting.")
    sys.exit(1)

# Extract signals
required_signals = ['I', 'SCG']
for sig in required_signals:
    if sig not in record.sig_name:
        print(f"Required signal '{sig}' not found.")
        sys.exit(1)

ecg = record.p_signal[:, record.sig_name.index('I')]
scg = record.p_signal[:, record.sig_name.index('SCG')]

# Limit samples and remove NaN values
ecg = ecg[:MAX_SAMPLES]
scg = scg[:MAX_SAMPLES]

# Remove NaN and infinite values
valid_mask = np.isfinite(ecg) & np.isfinite(scg)
ecg = ecg[valid_mask]
scg = scg[valid_mask]

print(f"Using {len(ecg)} valid samples after cleaning.")

# Apply advanced preprocessing
print("Applying advanced preprocessing...")
ecg = advanced_signal_preprocessing(ecg)
scg = advanced_signal_preprocessing(scg)

# ---------- ADVANCED SEQUENCE CREATION ----------
def create_sequences_with_augmentation(x, y, seq_len=SEQ_LEN, step_size=STEP_SIZE, 
                                     augment_prob=AUGMENT_PROB):
    """Create sequences with data augmentation"""
    xs, ys = [], []
    augmenter = AdvancedDataAugmentation()
    
    for i in range(0, len(x) - seq_len, step_size):
        x_seq = x[i:i+seq_len]
        y_seq = y[i:i+seq_len]
        
        # Original sequence
        xs.append(x_seq)
        ys.append(y_seq)
        
        # Augmented sequence
        if np.random.random() < augment_prob:
            x_aug = augmenter.apply_random_augmentation(x_seq.copy())
            y_aug = augmenter.apply_random_augmentation(y_seq.copy())
            xs.append(x_aug)
            ys.append(y_aug)
    
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X, Y = create_sequences_with_augmentation(ecg, scg)
X = X[:, :, np.newaxis]
Y = Y[:, :, np.newaxis]

print(f"Created {len(X)} sequences with augmentation (sequence length {SEQ_LEN})")

# ---------- ADVANCED DATA SPLITTING ----------
# Use stratified split to ensure balanced distribution
indices = np.arange(len(X))
X_temp, X_test, Y_temp, Y_test, idx_temp, idx_test = train_test_split(
    X, Y, indices, test_size=0.15, random_state=42, shuffle=True
)

X_train, X_val, Y_train, Y_val, _, _ = train_test_split(
    X_temp, Y_temp, idx_temp, test_size=0.176, random_state=42, shuffle=True  # 0.176 ≈ 0.15/0.85
)

print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# Create datasets and loaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         pin_memory=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                       pin_memory=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        pin_memory=False, num_workers=0)

# ---------- MODEL AND TRAINING SETUP ----------
model = ImprovedECGtoSCG(
    input_size=1,
    hidden_size=64,
    num_layers=3,
    output_size=1,
    dropout=0.2,
    use_attention=True,
    use_tcn=True
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Advanced loss function
criterion = AdvancedLoss(
    mse_weight=1.0,
    mae_weight=0.3,
    corr_weight=2.5,
    freq_weight=0.4,
    smooth_weight=0.15,
    phase_weight=0.3
)

# Advanced optimizer with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999)
)

# Learning rate scheduler with warm restarts
scheduler = CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=20,  # Initial restart period
    T_mult=2,  # Multiply restart period by this factor
    eta_min=1e-6
)

# ---------- ADVANCED TRAINING FUNCTIONS ----------
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def compute_comprehensive_metrics(y_true, y_pred):
    """Compute comprehensive evaluation metrics"""
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    
    # Basic metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # Correlation metrics
    pearson_corr, _ = pearsonr(y_true_flat, y_pred_flat)
    
    # Signal quality metrics
    signal_power = np.mean(y_true_flat**2)
    noise_power = np.mean((y_true_flat - y_pred_flat)**2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Frequency domain metrics
    y_true_fft = np.fft.fft(y_true_flat)
    y_pred_fft = np.fft.fft(y_pred_flat)
    spectral_correlation = np.corrcoef(np.abs(y_true_fft), np.abs(y_pred_fft))[0, 1]
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr,
        'snr': snr,
        'spectral_corr': spectral_correlation
    }

def evaluate_model(model, data_loader, criterion, device):
    """Comprehensive model evaluation"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    metrics = compute_comprehensive_metrics(all_targets, all_preds)
    metrics['loss'] = total_loss / len(data_loader)
    
    return metrics, all_preds, all_targets

# ---------- TRAINING LOOP ----------
print(f"Starting training for {EPOCHS} epochs...")
start_time = time.time()

# Training history
history = {
    'train_loss': [], 'val_loss': [],
    'train_rmse': [], 'val_rmse': [],
    'train_mae': [], 'val_mae': [],
    'train_r2': [], 'val_r2': [],
    'train_pearson': [], 'val_pearson': [],
    'train_snr': [], 'val_snr': [],
    'train_spectral_corr': [], 'val_spectral_corr': [],
    'learning_rate': []
}

best_val_loss = float('inf')
best_val_pearson = -1
patience_counter = 0
best_epoch = 0

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    # Training phase
    model.train()
    train_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch_idx, (x_batch, y_batch) in enumerate(progress_bar):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Apply mixup augmentation
        if np.random.random() < 0.5:
            x_batch, y_a, y_b, lam = mixup_data(x_batch, y_batch)
            outputs = model(x_batch)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        
        optimizer.step()
        
        train_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Memory cleanup
        if device.type == 'mps':
            torch.mps.empty_cache()
    
    # Calculate average training loss
    avg_train_loss = train_loss / num_batches
    
    # Evaluation phase
    train_metrics, _, _ = evaluate_model(model, train_loader, criterion, device)
    val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
    
    # Update learning rate
    scheduler.step()
    
    # Store metrics
    for key in ['loss', 'rmse', 'mae', 'r2', 'pearson', 'snr', 'spectral_corr']:
        history[f'train_{key}'].append(train_metrics[key])
        history[f'val_{key}'].append(val_metrics[key])
    
    history['learning_rate'].append(optimizer.param_groups[0]['lr'])
    
    # Calculate epoch time
    epoch_time = time.time() - epoch_start
    
    # Print epoch results
    print(f"\nEpoch {epoch+1}/{EPOCHS} ({epoch_time:.2f}s)")
    print(f"Train - Loss: {train_metrics['loss']:.4f}, RMSE: {train_metrics['rmse']:.4f}, "
          f"Pearson: {train_metrics['pearson']:.4f}, SNR: {train_metrics['snr']:.2f}")
    print(f"Val   - Loss: {val_metrics['loss']:.4f}, RMSE: {val_metrics['rmse']:.4f}, "
          f"Pearson: {val_metrics['pearson']:.4f}, SNR: {val_metrics['snr']:.2f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Model checkpointing based on validation Pearson correlation
    if val_metrics['pearson'] > best_val_pearson:
        best_val_pearson = val_metrics['pearson']
        best_val_loss = val_metrics['loss']
        best_epoch = epoch
        patience_counter = 0
        
        # Save best model
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_improved_model.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
            'train_metrics': train_metrics,
            'history': history
        }, checkpoint_path)
        
        print(f"✓ New best model saved! Pearson: {best_val_pearson:.4f}")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        break
    
    # Periodic checkpointing
    if (epoch + 1) % 20 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics
        }, checkpoint_path)
    
    # Memory cleanup
    if device.type == 'mps':
        torch.mps.empty_cache()
    
    print("-" * 80)

# Training summary
total_time = time.time() - start_time
print(f"\nTraining completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"Best model: Epoch {best_epoch+1}, Pearson: {best_val_pearson:.4f}, Loss: {best_val_loss:.4f}")

# Load best model for final evaluation
best_model_path = os.path.join(CHECKPOINT_DIR, "best_improved_model.pt")
checkpoint = torch.load(best_model_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# ---------- FINAL EVALUATION ----------
print("\n" + "="*50)
print("FINAL EVALUATION ON TEST SET")
print("="*50)

test_metrics, test_preds, test_labels = evaluate_model(model, test_loader, criterion, device)

print(f"\nTest Set Performance:")
print(f"MSE:                 {test_metrics['mse']:.6f}")
print(f"RMSE:                {test_metrics['rmse']:.6f}")
print(f"MAE:                 {test_metrics['mae']:.6f}")
print(f"R² Score:            {test_metrics['r2']:.6f}")
print(f"Pearson Correlation: {test_metrics['pearson']:.6f}")
print(f"SNR (dB):            {test_metrics['snr']:.6f}")
print(f"Spectral Correlation:{test_metrics['spectral_corr']:.6f}")

# Save detailed results
results_file = os.path.join(CHECKPOINT_DIR, "final_results.txt")
with open(results_file, 'w') as f:
    f.write("IMPROVED ECG TO SCG ESTIMATION RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Training Configuration:\n")
    f.write(f"- Epochs: {EPOCHS}\n")
    f.write(f"- Batch Size: {BATCH_SIZE}\n")
    f.write(f"- Learning Rate: {LEARNING_RATE}\n")
    f.write(f"- Sequence Length: {SEQ_LEN}\n")
    f.write(f"- Max Samples: {MAX_SAMPLES}\n")
    f.write(f"- Training Time: {total_time:.2f}s\n\n")
    
    f.write(f"Final Test Metrics:\n")
    for metric, value in test_metrics.items():
        f.write(f"- {metric.upper()}: {value:.6f}\n")
    
    f.write(f"\nBest Validation Performance:\n")
    f.write(f"- Epoch: {best_epoch+1}\n")
    f.write(f"- Pearson Correlation: {best_val_pearson:.6f}\n")
    f.write(f"- Loss: {best_val_loss:.6f}\n")

print(f"\nDetailed results saved to: {results_file}")

# ---------- ADVANCED VISUALIZATION ----------
print("\nCreating comprehensive visualizations...")

plt.style.use('seaborn-v0_8')
fig = plt.figure(figsize=(20, 16))

# 1. Signal comparison
ax1 = plt.subplot(3, 3, 1)
test_actual = test_labels[:5, :, 0].reshape(-1)
test_predicted = test_preds[:5, :, 0].reshape(-1)
time_axis = np.arange(len(test_actual)) / FS
plt.plot(time_axis[:1000], test_actual[:1000], 'b-', label='Actual SCG', linewidth=1.5, alpha=0.8)
plt.plot(time_axis[:1000], test_predicted[:1000], 'r--', label='Predicted SCG', linewidth=1.5, alpha=0.8)
plt.title('Signal Comparison (2 seconds)', fontsize=12, fontweight='bold')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Training history - Loss
ax2 = plt.subplot(3, 3, 2)
epochs_range = range(1, len(history['train_loss']) + 1)
plt.plot(epochs_range, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs_range, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
plt.axvline(x=best_epoch+1, color='green', linestyle='--', alpha=0.7, label=f'Best Model')
plt.title('Training Progress - Loss', fontsize=12, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# 3. Pearson correlation over time
ax3 = plt.subplot(3, 3, 3)
plt.plot(epochs_range, history['train_pearson'], 'b-', label='Training', linewidth=2)
plt.plot(epochs_range, history['val_pearson'], 'r-', label='Validation', linewidth=2)
plt.axvline(x=best_epoch+1, color='green', linestyle='--', alpha=0.7)
plt.title('Pearson Correlation Progress', fontsize=12, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Pearson Correlation')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. RMSE progress
ax4 = plt.subplot(3, 3, 4)
plt.plot(epochs_range, history['train_rmse'], 'b-', label='Training', linewidth=2)
plt.plot(epochs_range, history['val_rmse'], 'r-', label='Validation', linewidth=2)
plt.axvline(x=best_epoch+1, color='green', linestyle='--', alpha=0.7)
plt.title('RMSE Progress', fontsize=12, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. SNR progress
ax5 = plt.subplot(3, 3, 5)
plt.plot(epochs_range, history['train_snr'], 'b-', label='Training', linewidth=2)
plt.plot(epochs_range, history['val_snr'], 'r-', label='Validation', linewidth=2)
plt.axvline(x=best_epoch+1, color='green', linestyle='--', alpha=0.7)
plt.title('SNR Progress', fontsize=12, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('SNR (dB)')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Learning rate schedule
ax6 = plt.subplot(3, 3, 6)
plt.plot(epochs_range, history['learning_rate'], 'g-', linewidth=2)
plt.title('Learning Rate Schedule', fontsize=12, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.grid(True, alpha=0.3)
plt.yscale('log')

# 7. Scatter plot
ax7 = plt.subplot(3, 3, 7)
sample_size = min(3000, len(test_actual))
indices = np.random.choice(len(test_actual), sample_size, replace=False)
sampled_actual = test_actual[indices]
sampled_predicted = test_predicted[indices]

plt.scatter(sampled_actual, sampled_predicted, alpha=0.5, s=8, c='blue')
min_val, max_val = min(sampled_actual.min(), sampled_predicted.min()), max(sampled_actual.max(), sampled_predicted.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.title(f'Prediction Accuracy\n(r={test_metrics["pearson"]:.4f}, R²={test_metrics["r2"]:.4f})', 
          fontsize=12, fontweight='bold')
plt.xlabel('Actual SCG')
plt.ylabel('Predicted SCG')
plt.grid(True, alpha=0.3)
plt.axis('equal')

# 8. Frequency domain comparison
ax8 = plt.subplot(3, 3, 8)
freqs = np.fft.fftfreq(len(test_actual[:1000]), 1/FS)
magnitude_actual = np.abs(np.fft.fft(test_actual[:1000]))
magnitude_predicted = np.abs(np.fft.fft(test_predicted[:1000]))
valid_freq_idx = (freqs >= 0) & (freqs <= 25)  # Focus on 0-25 Hz

plt.semilogy(freqs[valid_freq_idx], magnitude_actual[valid_freq_idx], 'b-', 
             label='Actual', linewidth=2, alpha=0.8)
plt.semilogy(freqs[valid_freq_idx], magnitude_predicted[valid_freq_idx], 'r--', 
             label='Predicted', linewidth=2, alpha=0.8)
plt.title('Frequency Domain Comparison', fontsize=12, fontweight='bold')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True, alpha=0.3)

# 9. Error distribution
ax9 = plt.subplot(3, 3, 9)
errors = test_actual - test_predicted
plt.hist(errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.title(f'Prediction Error Distribution\n(μ={np.mean(errors):.4f}, σ={np.std(errors):.4f})', 
          fontsize=12, fontweight='bold')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, 'comprehensive_results.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

print(f"\nTraining completed successfully!")
print(f"Best model achieved Pearson correlation of {best_val_pearson:.4f}")
print(f"Test set Pearson correlation: {test_metrics['pearson']:.4f}")
print(f"All results saved in: {CHECKPOINT_DIR}/")

