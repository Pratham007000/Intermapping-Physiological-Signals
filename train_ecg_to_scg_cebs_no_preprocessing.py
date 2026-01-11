import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import wfdb
import os
from tqdm import tqdm
import psutil

# Data augmentation: Add small Gaussian noise to ECG signal
def add_noise(signal, noise_factor=0.005):
    noise = np.random.normal(0, noise_factor, len(signal))
    return signal + noise

# Enhanced Model with Attention and Residual Connections
class ECG2SCGModel(nn.Module):
    def __init__(self, input_size=200, hidden_size=256):
        super(ECG2SCGModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256 * input_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Softmax(dim=1)
        )
        
        # Projection layer to match dimensions for residual connection
        self.projection = nn.Conv1d(128, 256, kernel_size=1)
        
        self.fc1 = nn.Linear(256 * input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.relu(self.bn3(self.conv3(x2)))
        
        x_flat = x3.view(x3.size(0), -1)
        attn_weights = self.attention(x_flat)
        attn_weights = attn_weights.view(x3.size(0), x3.size(1), 1)
        x3 = x3 * attn_weights
        
        # Project x2 to match x3 dimensions for residual connection
        x2_projected = self.projection(x2)
        x3 = x3 + x2_projected
        x3 = x3.view(x3.size(0), -1)
        
        x3 = self.dropout(self.relu(self.fc1(x3)))
        out = self.fc2(x3)
        return out

# SNR Calculation
def calculate_snr(y_true, y_pred):
    signal_power = np.mean(y_true**2)
    noise_power = np.mean((y_true - y_pred)**2)
    return 10 * np.log10(signal_power / noise_power) if noise_power != 0 else float('inf')

# Load CEBS dataset (raw signals, no preprocessing)
record_name = 'b001'
if not os.path.exists(f'./cebs/{record_name}.hea') or not os.path.exists(f'./cebs/{record_name}.dat'):
    os.makedirs('./cebs', exist_ok=True)
    os.system(f"wget -P ./cebs/ https://physionet.org/files/cebsdb/1.0.0/{record_name}.hea")
    os.system(f"wget -P ./cebs/ https://physionet.org/files/cebsdb/1.0.0/{record_name}.dat")
else:
    print(f"CEBS database record '{record_name}' already exists.")
print("Successfully loaded the record.")

record = wfdb.rdrecord(f'./cebs/{record_name}')
print("Available signals:", record.sig_name)
ecg_signal = record.p_signal[:, record.sig_name.index('II')]
scg_signal = record.p_signal[:, record.sig_name.index('SCG')]

# Use a subset of the data
n_samples = 30000
ecg_signal = ecg_signal[:n_samples]
scg_signal = scg_signal[:n_samples]

# Create sequences
sequence_length = 200
step_size = 40
X, y = [], []
for i in range(0, len(ecg_signal) - sequence_length, step_size):
    X.append(add_noise(ecg_signal[i:i + sequence_length]))
    y.append(scg_signal[i:i + sequence_length])
X, y = np.array(X), np.array(y)
print(f"Total sequences: {len(X)} (sequence length {sequence_length}, step size {step_size})")

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
n_epochs = 200
batch_size = 64
k_folds = 5
patience = 25

# Cross-validation
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_tensor)):
    print(f"\nFold {fold + 1}/{k_folds}")
    
    train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
    val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = ECG2SCGModel(input_size=sequence_length).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = f"best_model_fold_{fold + 1}.pt"
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    val_pearsons = []
    val_snrs = []
    epochs = []
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Fold {fold + 1} Epoch {epoch + 1}/{n_epochs} [Train]"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        val_preds = np.concatenate(val_preds).flatten()
        val_targets = np.concatenate(val_targets).flatten()
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_pearson, _ = pearsonr(val_targets, val_preds)
        val_snr = calculate_snr(val_targets, val_preds)
        
        val_pearsons.append(val_pearson)
        val_snrs.append(val_snr)
        epochs.append(f"Epoch {epoch + 1}")
        
        print(f"Fold {fold + 1} Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, "
              f"Val Pearson: {val_pearson:.4f}, Val SNR: {val_snr:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model for fold {fold + 1} saved with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        memory = psutil.Process().memory_info().rss / 1024**3
        print(f"CPU RAM used: {memory:.2f} GB")
    
    # Generate line chart for this fold (training progress)
    print(f"\nTraining Progress Line Chart for Fold {fold + 1}:")
    chart_config = {
        "type": "line",
        "data": {
            "labels": epochs,
            "datasets": [
                {
                    "label": "Train Loss",
                    "data": train_losses,
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "backgroundColor": "rgba(75, 192, 192, 0.2)",
                    "fill": False
                },
                {
                    "label": "Val Loss",
                    "data": val_losses,
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "backgroundColor": "rgba(255, 99, 132, 0.2)",
                    "fill": False
                },
                {
                    "label": "Val Pearson",
                    "data": val_pearsons,
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "backgroundColor": "rgba(54, 162, 235, 0.2)",
                    "fill": False
                },
                {
                    "label": "Val SNR",
                    "data": val_snrs,
                    "borderColor": "rgba(153, 102, 255, 1)",
                    "backgroundColor": "rgba(153, 102, 255, 0.2)",
                    "fill": False
                }
            ]
        },
        "options": {
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "title": {
                        "display": True,
                        "text": "Metric Value"
                    }
                },
                "x": {
                    "title": {
                        "display": True,
                        "text": "Epoch"
                    }
                }
            },
            "plugins": {
                "title": {
                    "display": True,
                    "text": f"Training Progress for Fold {fold + 1}"
                }
            }
        }
    }
    print("```chartjs")
    print(chart_config)
    print("```")
    
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(batch_y.cpu().numpy())
    val_preds = np.concatenate(val_preds).flatten()
    val_targets = np.concatenate(val_targets).flatten()
    
    fold_mse = mean_squared_error(val_targets, val_preds)
    fold_rmse = np.sqrt(fold_mse)
    fold_mae = np.mean(np.abs(val_targets - val_preds))
    fold_r2 = r2_score(val_targets, val_preds)
    fold_pearson, _ = pearsonr(val_targets, val_preds)
    fold_snr = calculate_snr(val_targets, val_preds)
    
    fold_results.append({
        'MSE': fold_mse, 'RMSE': fold_rmse, 'MAE': fold_mae,
        'R2': fold_r2, 'PEARSON': fold_pearson, 'SNR': fold_snr
    })

# Average cross-validation metrics
avg_metrics = {}
for metric in fold_results[0].keys():
    avg_metrics[metric] = np.mean([fold[metric] for fold in fold_results])
print("\nAverage Cross-Validation Metrics:")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.6f}")

# Test set evaluation
test_size = int(0.2 * len(X_tensor))
train_val_X, test_X = X_tensor[:-test_size], X_tensor[-test_size:]
train_val_y, test_y = y_tensor[:-test_size], y_tensor[-test_size:]
test_dataset = TensorDataset(test_X, test_y)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\nEvaluating best model on test set...")
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        test_preds.append(outputs.cpu().numpy())
        test_targets.append(batch_y.cpu().numpy())
test_preds = np.concatenate(test_preds)
test_targets = np.concatenate(test_targets)

# Compute test set metrics (for the flattened arrays)
test_mse = mean_squared_error(test_targets.flatten(), test_preds.flatten())
test_rmse = np.sqrt(test_mse)
test_mae = np.mean(np.abs(test_targets.flatten() - test_preds.flatten()))
test_r2 = r2_score(test_targets.flatten(), test_preds.flatten())
test_pearson, _ = pearsonr(test_targets.flatten(), test_preds.flatten())
test_snr = calculate_snr(test_targets.flatten(), test_preds.flatten())

print("\nTest Set Metrics:")
print(f"MSE: {test_mse:.6f}")
print(f"RMSE: {test_rmse:.6f}")
print(f"MAE: {test_mae:.6f}")
print(f"R²: {test_r2:.6f}")
print(f"Pearson Correlation: {test_pearson:.6f}")
print(f"SNR: {test_snr:.6f}")

# Generate SCG plot for a single test sequence (200 samples)
print("\nGenerating SCG Plot for a Test Sequence...")
# Take the first sequence from the test set
sequence_idx = 0  # First sequence
actual_scg = test_targets[sequence_idx].tolist()  # 200 samples
predicted_scg = test_preds[sequence_idx].tolist()  # 200 samples

scg_chart_config = {
    "type": "line",
    "data": {
        "labels": list(range(sequence_length)),  # Sample indices 0 to 199
        "datasets": [
            {
                "label": "Actual SCG",
                "data": actual_scg,
                "borderColor": "rgba(75, 192, 192, 1)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "fill": False
            },
            {
                "label": "Predicted SCG",
                "data": predicted_scg,
                "borderColor": "rgba(255, 99, 132, 1)",
                "backgroundColor": "rgba(255, 99, 132, 0.2)",
                "fill": False
            }
        ]
    },
    "options": {
        "scales": {
            "y": {
                "title": {
                    "display": True,
                    "text": "SCG Signal Amplitude"
                }
            },
            "x": {
                "title": {
                    "display": True,
                    "text": "Sample Index"
                }
            }
        },
        "plugins": {
            "title": {
                "display": True,
                "text": "Actual vs Predicted SCG Signal (Test Sequence)"
            }
        }
    }
}
print("```chartjs")
print(scg_chart_config)
print("```")