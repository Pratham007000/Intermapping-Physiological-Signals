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
from simple_ecg_to_scg_model import SimplifiedECGtoSCG, SimplifiedLoss
from sklearn.model_selection import train_test_split
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings
warnings.filterwarnings('ignore')

# ---------- CPU-OPTIMIZED CONFIGURATION ----------
MAX_SAMPLES = 20000         # Reduced for CPU training
SEQ_LEN = 64                # Shorter sequences for faster CPU processing
STEP_SIZE = 16              # Smaller step size
BATCH_SIZE = 8              # Smaller batch size for CPU
EPOCHS = 50                 # Fewer epochs for quicker results
LEARNING_RATE = 0.002       # Slightly higher LR for faster convergence
WEIGHT_DECAY = 1e-4         # L2 regularization
EARLY_STOPPING_PATIENCE = 15
CHECKPOINT_DIR = "improved_checkpoints_cpu"
FS = 500                    # Sampling frequency

# Training parameters optimized for CPU
GRADIENT_CLIP = 0.5
MIXUP_ALPHA = 0.1
AUGMENT_PROB = 0.5

# Force CPU usage
device = torch.device("cpu")
print(f"Using device: {device}")

# Set number of threads for CPU optimization
torch.set_num_threads(4)  # Adjust based on your CPU cores
print(f"Using {torch.get_num_threads()} CPU threads")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Create checkpoint directory
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

# ---------- SIMPLIFIED DATA AUGMENTATION FOR CPU ----------
class CPUOptimizedAugmentation:
    def __init__(self, fs=FS):
        self.fs = fs
        
    def add_noise(self, signal, noise_level=0.03):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level, signal.shape)
        return signal + noise
    
    def time_shift(self, signal, max_shift=5):
        """Random time shifting"""
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(signal, shift)
    
    def amplitude_scale(self, signal, scale_range=(0.9, 1.1)):
        """Random amplitude scaling"""
        scale = np.random.uniform(*scale_range)
        return signal * scale
    
    def apply_augmentation(self, signal, prob=0.5):
        """Apply simple augmentations"""
        if np.random.random() < prob:
            # Randomly choose one augmentation
            aug_type = np.random.choice([0, 1, 2])
            if aug_type == 0:
                signal = self.add_noise(signal)
            elif aug_type == 1:
                signal = self.time_shift(signal)
            else:
                signal = self.amplitude_scale(signal)
        return signal

# ---------- EFFICIENT PREPROCESSING FOR CPU ----------
def efficient_preprocessing(data_signal, fs=FS):
    """Efficient preprocessing for CPU"""
    # Simple outlier removal
    q75, q25 = np.percentile(data_signal, [75, 25])
    iqr = q75 - q25
    data_signal = np.clip(data_signal, q25 - 1.5*iqr, q75 + 1.5*iqr)
    
    # Basic filtering
    # High-pass filter
    sos_hp = signal.butter(2, 0.5, 'high', fs=fs, output='sos')
    data_signal = signal.sosfilt(sos_hp, data_signal)
    
    # Band-pass filter
    sos_bp = signal.butter(4, [0.5, 40], 'band', fs=fs, output='sos')
    data_signal = signal.sosfilt(sos_bp, data_signal)
    
    # Robust normalization
    median = np.median(data_signal)
    mad = np.median(np.abs(data_signal - median))
    data_signal = (data_signal - median) / (1.4826 * mad + 1e-8)
    
    return data_signal

# ---------- DATA LOADING ----------
def load_data():
    """Load CEBS database"""
    db_dir = "cebsdb"
    record_name = "b001"
    
    try:
        record = wfdb.rdrecord(record_name, pn_dir=db_dir)
        print("Successfully loaded the record.")
        print("Available signals:", record.sig_name)
        return record
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

record = load_data()
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

valid_mask = np.isfinite(ecg) & np.isfinite(scg)
ecg = ecg[valid_mask]
scg = scg[valid_mask]

print(f"Using {len(ecg)} valid samples after cleaning.")

# Apply preprocessing
print("Applying preprocessing...")
ecg = efficient_preprocessing(ecg)
scg = efficient_preprocessing(scg)

# ---------- SEQUENCE CREATION ----------
def create_sequences(x, y, seq_len=SEQ_LEN, step_size=STEP_SIZE):
    """Create sequences efficiently"""
    xs, ys = [], []
    augmenter = CPUOptimizedAugmentation()
    
    for i in range(0, len(x) - seq_len, step_size):
        x_seq = x[i:i+seq_len]
        y_seq = y[i:i+seq_len]
        
        # Original sequence
        xs.append(x_seq)
        ys.append(y_seq)
        
        # Augmented sequence (50% chance)
        if np.random.random() < AUGMENT_PROB:
            x_aug = augmenter.apply_augmentation(x_seq.copy())
            y_aug = augmenter.apply_augmentation(y_seq.copy())
            xs.append(x_aug)
            ys.append(y_aug)
    
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X, Y = create_sequences(ecg, scg)
X = X[:, :, np.newaxis]
Y = Y[:, :, np.newaxis]

print(f"Created {len(X)} sequences (sequence length {SEQ_LEN})")

# ---------- DATA SPLITTING ----------
indices = np.arange(len(X))
X_temp, X_test, Y_temp, Y_test, _, _ = train_test_split(
    X, Y, indices, test_size=0.15, random_state=42, shuffle=True
)

X_train, X_val, Y_train, Y_val, _, _ = train_test_split(
    X_temp, Y_temp, indices[:-len(X_test)], test_size=0.176, random_state=42, shuffle=True
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

# No workers to avoid multiprocessing issues
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------- MODEL SETUP ----------
# Simplified model for CPU efficiency
model = SimplifiedECGtoSCG(
    input_size=1,
    hidden_size=32,      # Reduced hidden size
    num_layers=2,        # Fewer layers
    output_size=1,
    dropout=0.3,
    use_attention=True   # Keep attention for accuracy
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Simplified loss function for CPU
criterion = SimplifiedLoss(
    mse_weight=1.0,
    mae_weight=0.3,
    corr_weight=1.5,
    smooth_weight=0.1
)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999)
)

# Learning rate scheduler
scheduler = CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10,
    T_mult=2,
    eta_min=1e-5
)

# ---------- TRAINING FUNCTIONS ----------
def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics"""
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # Avoid NaN in correlation
    try:
        pearson_corr, _ = pearsonr(y_true_flat, y_pred_flat)
        if np.isnan(pearson_corr):
            pearson_corr = 0.0
    except:
        pearson_corr = 0.0
    
    # SNR calculation
    signal_power = np.mean(y_true_flat**2)
    noise_power = np.mean((y_true_flat - y_pred_flat)**2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr,
        'snr': snr
    }

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model"""
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
    
    metrics = compute_metrics(all_targets, all_preds)
    metrics['loss'] = total_loss / len(data_loader)
    
    return metrics, all_preds, all_targets

# ---------- TRAINING LOOP ----------
print(f"Starting CPU training for {EPOCHS} epochs...")
start_time = time.time()

# Training history
history = {
    'train_loss': [], 'val_loss': [],
    'train_rmse': [], 'val_rmse': [],
    'train_r2': [], 'val_r2': [],
    'train_pearson': [], 'val_pearson': [],
    'train_snr': [], 'val_snr': [],
    'learning_rate': []
}

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
        
        # Simple mixup (reduced for CPU efficiency)
        if np.random.random() < 0.3:  # Reduced probability
            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
            batch_size = x_batch.size(0)
            index = torch.randperm(batch_size)
            
            mixed_x = lam * x_batch + (1 - lam) * x_batch[index]
            y_a, y_b = y_batch, y_batch[index]
            
            outputs = model(mixed_x)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
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
    
    # Calculate average training loss
    avg_train_loss = train_loss / num_batches
    
    # Evaluation phase
    train_metrics, _, _ = evaluate_model(model, train_loader, criterion, device)
    val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
    
    # Update learning rate
    scheduler.step()
    
    # Store metrics
    for key in ['loss', 'rmse', 'r2', 'pearson', 'snr']:
        history[f'train_{key}'].append(train_metrics[key])
        history[f'val_{key}'].append(val_metrics[key])
    
    history['learning_rate'].append(optimizer.param_groups[0]['lr'])
    
    # Calculate epoch time
    epoch_time = time.time() - epoch_start
    
    # Print epoch results
    print(f"\nEpoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s)")
    print(f"Train - Loss: {train_metrics['loss']:.4f}, RMSE: {train_metrics['rmse']:.4f}, "
          f"Pearson: {train_metrics['pearson']:.4f}")
    print(f"Val   - Loss: {val_metrics['loss']:.4f}, RMSE: {val_metrics['rmse']:.4f}, "
          f"Pearson: {val_metrics['pearson']:.4f}")
    
    # Model checkpointing
    if val_metrics['pearson'] > best_val_pearson:
        best_val_pearson = val_metrics['pearson']
        best_epoch = epoch
        patience_counter = 0
        
        # Save best model
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_cpu_model.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'history': history
        }, checkpoint_path)
        
        print(f"✓ New best model saved! Pearson: {best_val_pearson:.4f}")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        break
    
    print("-" * 60)

# Training summary
total_time = time.time() - start_time
print(f"\nTraining completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print(f"Best model: Epoch {best_epoch+1}, Pearson: {best_val_pearson:.4f}")

# Load best model for final evaluation
best_model_path = os.path.join(CHECKPOINT_DIR, "best_cpu_model.pt")
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

# Save results
results_file = os.path.join(CHECKPOINT_DIR, "cpu_results.txt")
with open(results_file, 'w') as f:
    f.write("IMPROVED ECG TO SCG ESTIMATION RESULTS (CPU)\n")
    f.write("="*50 + "\n\n")
    f.write(f"Training Configuration:\n")
    f.write(f"- Device: {device}\n")
    f.write(f"- Epochs: {EPOCHS}\n")
    f.write(f"- Batch Size: {BATCH_SIZE}\n")
    f.write(f"- Learning Rate: {LEARNING_RATE}\n")
    f.write(f"- Sequence Length: {SEQ_LEN}\n")
    f.write(f"- Max Samples: {MAX_SAMPLES}\n")
    f.write(f"- Training Time: {total_time:.1f}s\n\n")
    
    f.write(f"Final Test Metrics:\n")
    for metric, value in test_metrics.items():
        f.write(f"- {metric.upper()}: {value:.6f}\n")
    
    f.write(f"\nBest Validation Performance:\n")
    f.write(f"- Epoch: {best_epoch+1}\n")
    f.write(f"- Pearson Correlation: {best_val_pearson:.6f}\n")

print(f"\nDetailed results saved to: {results_file}")

# ---------- VISUALIZATION ----------
print("\nCreating visualizations...")

plt.figure(figsize=(15, 10))

# 1. Signal comparison
plt.subplot(2, 3, 1)
test_actual = test_labels[:3, :, 0].reshape(-1)
test_predicted = test_preds[:3, :, 0].reshape(-1)
time_axis = np.arange(len(test_actual)) / FS
plt.plot(time_axis[:500], test_actual[:500], 'b-', label='Actual SCG', linewidth=1.5)
plt.plot(time_axis[:500], test_predicted[:500], 'r--', label='Predicted SCG', linewidth=1.5)
plt.title('Signal Comparison (1 second)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Training history - Loss
plt.subplot(2, 3, 2)
epochs_range = range(1, len(history['train_loss']) + 1)
plt.plot(epochs_range, history['train_loss'], 'b-', label='Training Loss')
plt.plot(epochs_range, history['val_loss'], 'r-', label='Validation Loss')
plt.axvline(x=best_epoch+1, color='green', linestyle='--', alpha=0.7)
plt.title('Training Progress - Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Pearson correlation
plt.subplot(2, 3, 3)
plt.plot(epochs_range, history['train_pearson'], 'b-', label='Training')
plt.plot(epochs_range, history['val_pearson'], 'r-', label='Validation')
plt.axvline(x=best_epoch+1, color='green', linestyle='--', alpha=0.7)
plt.title('Pearson Correlation Progress')
plt.xlabel('Epochs')
plt.ylabel('Pearson Correlation')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. RMSE progress
plt.subplot(2, 3, 4)
plt.plot(epochs_range, history['train_rmse'], 'b-', label='Training')
plt.plot(epochs_range, history['val_rmse'], 'r-', label='Validation')
plt.axvline(x=best_epoch+1, color='green', linestyle='--', alpha=0.7)
plt.title('RMSE Progress')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Scatter plot
plt.subplot(2, 3, 5)
sample_size = min(1000, len(test_actual))
indices = np.random.choice(len(test_actual), sample_size, replace=False)
sampled_actual = test_actual[indices]
sampled_predicted = test_predicted[indices]

plt.scatter(sampled_actual, sampled_predicted, alpha=0.6, s=10)
min_val = min(sampled_actual.min(), sampled_predicted.min())
max_val = max(sampled_actual.max(), sampled_predicted.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.title(f'Prediction Accuracy\n(r={test_metrics["pearson"]:.3f}, R²={test_metrics["r2"]:.3f})')
plt.xlabel('Actual SCG')
plt.ylabel('Predicted SCG')
plt.grid(True, alpha=0.3)

# 6. Learning rate
plt.subplot(2, 3, 6)
plt.plot(epochs_range, history['learning_rate'], 'g-')
plt.title('Learning Rate Schedule')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, 'cpu_training_results.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\nCPU Training completed successfully!")
print(f"Best validation Pearson correlation: {best_val_pearson:.4f}")
print(f"Test set Pearson correlation: {test_metrics['pearson']:.4f}")
print(f"Improvement over original model expected!")
print(f"All results saved in: {CHECKPOINT_DIR}/")

