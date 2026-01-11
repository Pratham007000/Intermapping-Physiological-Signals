import wfdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
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
import psutil
import gc

# ---------- CONFIGURATION ----------
MAX_SAMPLES = 30000      # Number of samples
SEQ_LEN = 200            # Sequence length
STEP_SIZE = 40           # Step size for overlap
BATCH_SIZE = 8           # Batch size for CPU
ACCUMULATION_STEPS = 4   # Gradient accumulation
EPOCHS = 150             # Epochs
LEARNING_RATE = 0.0001   # Lower learning rate for stability
WARMUP_EPOCHS = 20       # Extended warm-up
EARLY_STOPPING_PATIENCE = 25  # Patience
CHECKPOINT_DIR = "checkpoints_cpu_high_accuracy"  # Directory for checkpoints
K_FOLDS = 5              # Cross-validation folds
FS = 500                 # Sampling frequency (Hz)

# Create checkpoint directory
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

# ---------- DEVICE SELECT ----------
device = torch.device("cpu")
print(f"Using device: {device}")

# Memory monitoring
def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"CPU RAM used: {mem_info.rss / 1024**3:.2f} GB")

# ---------- MODEL DEFINITION ----------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size = x.size(0)
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output), attention_weights

class ECGtoSCG_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, output_size=1, 
                 bidirectional=True, dropout=0.4, use_layernorm=True):
        super(ECGtoSCG_LSTM, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_layernorm = use_layernorm
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList() if use_layernorm else None
        
        lstm_input_size = input_size
        for i in range(num_layers):
            self.lstm_layers.append(
                nn.LSTM(lstm_input_size, hidden_size, 1, batch_first=True,
                        bidirectional=bidirectional, dropout=0)
            )
            if use_layernorm:
                self.ln_layers.append(
                    nn.LayerNorm(hidden_size * 2 if bidirectional else hidden_size)
                )
            lstm_input_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_size * 2 if bidirectional else hidden_size, num_heads=8)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer with scaling to handle raw SCG amplitude range
        self.linear = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        
        # Residual projection
        self.residual_proj = nn.Linear(input_size, output_size) if input_size != output_size else None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('tanh'))
            elif 'weight' in name and param.dim() == 1:
                nn.init.uniform_(param, -0.05, 0.05)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        residual = x
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN or Inf detected in input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        for i in range(self.num_layers):
            x, _ = self.lstm_layers[i](x)
            if self.use_layernorm:
                x = self.ln_layers[i](x)
            if i < self.num_layers - 1:
                x = self.dropout(x)
        
        x, attention_weights = self.attention(x)
        x = self.dropout(x)
        out = self.linear(x)
        
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        out = out + residual
        
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("Warning: NaN or Inf detected in model output")
            out = torch.nan_to_num(out, nan=0.0, posinf=5.0, neginf=-5.0)  # Adjusted for raw SCG range
        
        return out

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
    sys.exit(1)

try:
    record = wfdb.rdrecord(record_name, pn_dir="cebsdb")
    print("Successfully loaded the record.")
except Exception as e:
    print(f"Error reading record: {str(e)}")
    sys.exit(1)

print("Available signals:", record.sig_name)

required_signals = ['I', 'SCG']
for sig in required_signals:
    if sig not in record.sig_name:
        print(f"Required signal '{sig}' not found.")
        sys.exit(1)

ecg = record.p_signal[:, record.sig_name.index('I')]
scg = record.p_signal[:, record.sig_name.index('SCG')]

# Limit samples
ecg = ecg[:MAX_SAMPLES]
scg = scg[:MAX_SAMPLES]
print(f"Using {len(ecg)} samples.")

# Check for NaN in raw data
if np.isnan(ecg).any() or np.isnan(scg).any():
    print("Warning: NaN detected in raw ECG or SCG data")
    ecg = np.nan_to_num(ecg, nan=0.0)
    scg = np.nan_to_num(scg, nan=0.0)

# ---------- DATA PREPROCESSING ----------
def wavelet_denoise(signal_data, wavelet='db8', level=5, threshold_type='soft'):
    """Adaptive wavelet denoising with reduced level to preserve more signal details"""
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(signal_data))) / np.log(level + 2)
    coeffs[1:] = [pywt.threshold(c, threshold, mode=threshold_type) for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs, wavelet)
    if np.isnan(denoised).any() or np.isinf(denoised).any():
        print("Warning: NaN or Inf in wavelet denoising")
        denoised = np.nan_to_num(denoised, nan=0.0)
    return denoised

def filter_signal(signal_data, fs=FS, lowcut=0.5, highcut=40.0, order=4):
    """Apply bandpass filter with reduced order to avoid over-filtering"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, signal_data)
    if np.isnan(filtered).any() or np.isinf(filtered).any():
        print("Warning: NaN or Inf in bandpass filter")
        filtered = np.nan_to_num(filtered, nan=0.0)
    return filtered

def robust_normalize(signal, percentile_clip=99.8, target_std=1.0, max_clip=5.0):
    """Modified normalization to handle raw SCG amplitudes"""
    threshold = np.percentile(np.abs(signal), percentile_clip)
    signal = np.clip(signal, -threshold, threshold)
    signal = np.clip(signal, -max_clip, max_clip)
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        print("Warning: Standard deviation is zero in normalization")
        std = 1.0
    normalized = (signal - mean) / (std + 1e-10) * target_std
    if np.isnan(normalized).any() or np.isinf(normalized).any():
        print("Warning: NaN or Inf in normalization")
        normalized = np.nan_to_num(normalized, nan=0.0)
    return normalized

def augment_signal(signal, max_shift=10, noise_std=0.05, scale_range=(0.9, 1.1)):
    """Adjusted data augmentation to avoid distorting raw SCG"""
    if np.random.rand() < 0.5:
        shift = np.random.randint(-max_shift, max_shift)
        signal = np.roll(signal, shift)
    if np.random.rand() < 0.5:
        signal = signal + np.random.normal(0, noise_std, signal.shape)
    if np.random.rand() < 0.5:
        scale = np.random.uniform(scale_range[0], scale_range[1])
        signal = signal * scale
    if np.isnan(signal).any() or np.isinf(signal).any():
        print("Warning: NaN or Inf in augmentation")
        signal = np.nan_to_num(signal, nan=0.0)
    return signal

# Apply preprocessing
print("Applying preprocessing...")
# ECG preprocessing: Denoise and filter, then normalize
ecg = wavelet_denoise(ecg, wavelet='db8', level=5)
ecg = filter_signal(ecg, fs=FS, lowcut=0.5, highcut=40.0, order=4)
ecg = robust_normalize(ecg, target_std=1.0, max_clip=1.5)  # Adjusted for ECG range

# SCG: Use raw data, only apply minimal cleaning and scaling
scg = robust_normalize(scg, target_std=2.0, max_clip=5.0)  # Scale to preserve raw SCG range

def create_sequences(x, y, seq_len=SEQ_LEN, step_size=STEP_SIZE):
    xs, ys = [], []
    for i in range(0, len(x) - seq_len, step_size):
        xs.append(augment_signal(x[i:i+seq_len]))
        ys.append(y[i:i+seq_len])
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    if np.isnan(xs).any() or np.isnan(ys).any():
        print("Warning: NaN in sequences")
        xs = np.nan_to_num(xs, nan=0.0)
        ys = np.nan_to_num(ys, nan=0.0)
    return xs, ys

X, Y = create_sequences(ecg, scg, seq_len=SEQ_LEN, step_size=STEP_SIZE)
X = X[:, :, np.newaxis]
Y = Y[:, :, np.newaxis]
print(f"Total sequences: {len(X)} (sequence length {SEQ_LEN}, step size {STEP_SIZE})")

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

if torch.isnan(X_tensor).any() or torch.isnan(Y_tensor).any():
    print("Warning: NaN in input tensors")
    X_tensor = torch.nan_to_num(X_tensor, nan=0.0)
    Y_tensor = torch.nan_to_num(Y_tensor, nan=0.0)

# Create dataset
full_dataset = TensorDataset(X_tensor, Y_tensor)

# ---------- METRICS AND LOSS ----------
def correlation_loss(y_true, y_pred):
    y_true_mean = torch.mean(y_true, dim=1, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=1, keepdim=True)
    num = torch.sum((y_true - y_true_mean) * (y_pred - y_pred_mean), dim=1)
    denom = torch.sqrt(torch.sum((y_true - y_true_mean)**2, dim=1) * 
                      torch.sum((y_pred - y_pred_mean)**2, dim=1) + 1e-8)
    corr = num / denom
    corr = torch.where(torch.isnan(corr), torch.zeros_like(corr), corr)
    return -torch.mean(corr)

def smoothness_loss(y_pred):
    diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    return torch.mean(torch.abs(diff))

class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=1.0, corr_weight=2.5, smooth_weight=0.01):  # Reduced smooth_weight
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.corr_weight = corr_weight
        self.mse_weight = mse_weight
        self.smooth_weight = smooth_weight
    
    def forward(self, y_pred, y_true):
        mse = self.mse_loss(y_pred, y_true)
        corr = correlation_loss(y_true, y_pred)
        smooth = smoothness_loss(y_pred)
        if torch.isnan(mse) or torch.isnan(corr) or torch.isnan(smooth):
            print("Warning: NaN in loss components")
            mse = torch.nan_to_num(mse, nan=0.0)
            corr = torch.nan_to_num(corr, nan=0.0)
            smooth = torch.nan_to_num(smooth, nan=0.0)
        return self.mse_weight * mse + self.corr_weight * corr + self.smooth_weight * smooth

def compute_snr(y_true, y_pred):
    signal_power = np.mean(y_true**2)
    noise_power = np.mean((y_true - y_pred)**2)
    return 10 * np.log10(signal_power / (noise_power + 1e-10))

def compute_metrics(y_true, y_pred):
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    
    if np.isnan(y_true_flat).any() or np.isnan(y_pred_flat).any():
        print("Warning: NaN in metrics computation")
        y_true_flat = np.nan_to_num(y_true_flat, nan=0.0)
        y_pred_flat = np.nan_to_num(y_pred_flat, nan=0.0)
    
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    pearson_corr, _ = pearsonr(y_true_flat, y_pred_flat)
    snr = compute_snr(y_true_flat, y_pred_flat)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr,
        'snr': snr
    }

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
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

def save_checkpoint(model, optimizer, epoch, val_loss, metrics, filename, fold=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'metrics': metrics,
        'model_config': {
            'input_size': 1,
            'hidden_size': 64,
            'num_layers': 3,
            'output_size': 1,
            'bidirectional': True,
            'dropout': 0.4,
            'use_layernorm': True
        }
    }
    if fold is not None:
        checkpoint['fold'] = fold
    torch.save(checkpoint, filename)

# ---------- CROSS-VALIDATION TRAINING ----------
kfold = KFold(n_splits=K_FOLDS, shuffle=False)
fold_results = []

print(f"Starting {K_FOLDS}-fold cross-validation training for {EPOCHS} epochs...")
start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    print(f"\nFold {fold+1}/{K_FOLDS}")
    
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False, 
                             pin_memory=False, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, 
                           pin_memory=False, num_workers=0)
    
    model = ECGtoSCG_LSTM(use_layernorm=True).to(device)
    criterion = CombinedLoss(mse_weight=1.0, corr_weight=2.5, smooth_weight=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / WARMUP_EPOCHS)
    
    train_losses = []
    val_losses = []
    history = {
        'train_loss': [], 'val_loss': [],
        'train_rmse': [], 'val_rmse': [],
        'train_mae': [], 'val_mae': [],
        'train_r2': [], 'val_r2': [],
        'train_pearson': [], 'val_pearson': [],
        'train_snr': [], 'val_snr': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{EPOCHS} [Train]")
        
        optimizer.zero_grad()
        accumulation_count = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(progress_bar):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            if torch.rand(1).item() < 0.5:
                noise = torch.normal(0, 0.05, size=x_batch.shape).to(device)
                x_batch = x_batch + noise
            if torch.rand(1).item() < 0.5:
                shift = int(torch.randint(-10, 10, (1,)).item())
                x_batch = torch.roll(x_batch, shifts=shift, dims=1)
            
            with torch.amp.autocast('cpu'):
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch) / ACCUMULATION_STEPS
            loss.backward()
            
            accumulation_count += 1
            
            if accumulation_count == ACCUMULATION_STEPS or batch_idx == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad()
                accumulation_count = 0
            
            epoch_loss += loss.item() * ACCUMULATION_STEPS
            progress_bar.set_postfix(loss=f"{loss.item() * ACCUMULATION_STEPS:.4f}")
            
            del x_batch, y_batch, outputs, loss
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_losses.append(val_loss)
        
        train_metrics, _, _ = evaluate_model(model, train_loader, criterion, device)
        
        for key in history:
            history[key].append(train_metrics[key.split('_')[1]] if 'train' in key else val_metrics[key.split('_')[1]])
        
        print(f"Fold {fold+1} Epoch {epoch+1}/{EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val RMSE: {val_metrics['rmse']:.4f}, Val Pearson: {val_metrics['pearson']:.4f}, "
              f"Val SNR: {val_metrics['snr']:.4f}")
        
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_fold_{fold+1}.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, val_metrics, checkpoint_path, fold=fold+1)
            print(f"New best model for fold {fold+1} saved with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold+1}_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, val_metrics, checkpoint_path, fold=fold+1)
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs in fold {fold+1}")
            break
        
        print_memory_usage()
        gc.collect()
    
    fold_results.append({
        'fold': fold + 1,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch + 1,
        'val_metrics': val_metrics,
        'history': history
    })
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 2, 1)
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss', color='blue')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss', color='orange')
    plt.axvline(x=best_epoch+1, color='red', linestyle='--', label=f'Best Model (Epoch {best_epoch+1})')
    plt.title(f"Fold {fold+1}: Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(3, 2, 2)
    plt.plot(epochs_range, history['train_rmse'], label='Training RMSE', color='blue')
    plt.plot(epochs_range, history['val_rmse'], label='Validation RMSE', color='orange')
    plt.title(f"Fold {fold+1}: RMSE Over Training")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.legend()
    
    plt.subplot(3, 2, 3)
    plt.plot(epochs_range, history['train_pearson'], label='Training Pearson r', color='blue')
    plt.plot(epochs_range, history['val_pearson'], label='Validation Pearson r', color='orange')
    plt.title(f"Fold {fold+1}: Pearson Correlation Over Training")
    plt.xlabel("Epochs")
    plt.ylabel("Correlation Coefficient")
    plt.legend()
    
    plt.subplot(3, 2, 4)
    plt.plot(epochs_range, history['train_snr'], label='Training SNR', color='blue')
    plt.plot(epochs_range, history['val_snr'], label='Validation SNR', color='orange')
    plt.title(f"Fold {fold+1}: SNR Over Training")
    plt.xlabel("Epochs")
    plt.ylabel("SNR (dB)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, f"fold_{fold+1}_training_results.png"))
    plt.close()

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# ---------- EVALUATE BEST MODEL ON TEST SET ----------
best_fold = min(fold_results, key=lambda x: x['best_val_loss'])
best_model_path = os.path.join(CHECKPOINT_DIR, f"best_model_fold_{best_fold['fold']}.pt")
checkpoint = torch.load(best_model_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

n_samples = len(X)
test_size = int(0.15 * n_samples)
test_idx = slice(n_samples - test_size, n_samples)
test_dataset = TensorDataset(X_tensor[test_idx], Y_tensor[test_idx])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        pin_memory=False, num_workers=0)

print("\nEvaluating best model on test set...")
test_metrics, test_preds, test_labels = evaluate_model(model, test_loader, criterion, device)

print("\nTest Set Metrics:")
print(f"MSE: {test_metrics['mse']:.6f}")
print(f"RMSE: {test_metrics['rmse']:.6f}")
print(f"MAE: {test_metrics['mae']:.6f}")
print(f"R²: {test_metrics['r2']:.6f}")
print(f"Pearson Correlation: {test_metrics['pearson']:.6f}")
print(f"SNR: {test_metrics['snr']:.6f}")

# Average cross-validation metrics
avg_val_metrics = {}
for key in ['mse', 'rmse', 'mae', 'r2', 'pearson', 'snr']:
    avg_val_metrics[key] = np.mean([fold['val_metrics'][key] for fold in fold_results])
print("\nAverage Cross-Validation Metrics:")
for key, value in avg_val_metrics.items():
    print(f"{key.upper()}: {value:.6f}")

# ---------- PLOTTING ----------
plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
actual = test_labels[:300, :, 0].reshape(-1)
predicted = test_preds[:300, :, 0].reshape(-1)
plt.plot(actual[:500], label='Actual SCG', color='blue')
plt.plot(predicted[:500], label='Predicted SCG', color='orange', linestyle='--')
plt.title("Actual vs Predicted SCG (First 500 Timepoints)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(2, 1, 2)
test_preds_flat = test_preds.reshape(-1)
test_labels_flat = test_labels.reshape(-1)
sample_size = min(2000, len(test_preds_flat))
indices = np.random.choice(len(test_preds_flat), sample_size, replace=False)
sampled_preds = test_preds_flat[indices]
sampled_labels = test_labels_flat[indices]

plt.scatter(sampled_labels, sampled_preds, alpha=0.5)
plt.plot([-5, 5], [-5, 5], 'r--')  # Adjusted range for raw SCG
plt.title(f"Predicted vs Actual SCG Values (r={test_metrics['pearson']:.4f}, R²={test_metrics['r2']:.4f})")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, "test_results.png"))
plt.show()

# Save cross-validation summary
with open(os.path.join(CHECKPOINT_DIR, "cross_validation_summary.txt"), 'w') as f:
    for fold in fold_results:
        f.write(f"Fold {fold['fold']} - Best Epoch: {fold['best_epoch']}, Best Val Loss: {fold['best_val_loss']:.6f}\n")
        for key, value in fold['val_metrics'].items():
            f.write(f"  {key.upper()}: {value:.6f}\n")
    f.write("\nAverage Cross-Validation Metrics:\n")
    for key, value in avg_val_metrics.items():
        f.write(f"{key.upper()}: {value:.6f}\n")

print("\nTraining visualizations and summary saved to:", CHECKPOINT_DIR)