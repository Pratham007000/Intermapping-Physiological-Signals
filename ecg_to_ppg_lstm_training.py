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

# Load ECG and PPG data from realistic CSV file
def load_data():
    """Load data from realistic_ecg_ppg_data.csv or generate if not found"""
    try:
        data = pd.read_csv("realistic_ecg_ppg_data.csv")
        ecg = data["ECG Amplitude"].values
        ppg = data["PPG Amplitude"].values
        print(f"Loaded {len(ecg)} samples from realistic_ecg_ppg_data.csv")
        return ecg, ppg
    except FileNotFoundError:
        print("realistic_ecg_ppg_data.csv not found. Please run generate_realistic_ecg_data.py first.")
        exit()

ecg, ppg = load_data()
assert ecg.shape == ppg.shape

# Normalize
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
# Create model directory if it doesn't exist
model_dir = "model_checkpoints"
os.makedirs(model_dir, exist_ok=True)

# Initialize model, loss, optimizer
model = ECGtoPPG_LSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=5, factor=0.5
)

# Define number of epochs and early stopping parameters
num_epochs = 50
early_stopping_patience = 10

# Lists to store metrics for training monitoring
train_losses = []
val_losses = []

# Variables for early stopping
best_val_loss = float('inf')
early_stopping_counter = 0
best_model_path = os.path.join(model_dir, "best_lstm_model.pth")

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
        
        # Print progress every 10 batches
        if batch_count % 10 == 0:
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
    
    # Concatenate batch predictions and calculate validation metrics
    val_preds_array = np.concatenate(all_val_preds)
    val_labels_array = np.concatenate(all_val_labels)
    val_metrics = calculate_metrics(val_labels_array, val_preds_array)
    
    # Store basic metrics for monitoring
    # (Detailed metrics only used for epoch reporting)
    
    # Step the scheduler based on validation loss
    scheduler.step(avg_val_loss)
    
    # Check for early stopping and model saving
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
        
        # Save the best model (only state dict for compatibility)
        torch.save(model.state_dict(), best_model_path)
        
        # Save additional metadata separately for reference
        metadata_path = os.path.join(model_dir, "best_model_metadata.pth")
        torch.save({
            'epoch': epoch,
            'val_loss': best_val_loss,
            'train_loss': avg_train_loss,
            'val_metrics': val_metrics
        }, metadata_path)
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

# Load the best model for evaluation
try:
    # Try loading the state dict directly (new format)
    model.load_state_dict(torch.load(best_model_path))
    print("Loaded best model state dict successfully")
    
    # Try to load metadata if available
    metadata_path = os.path.join(model_dir, "best_model_metadata.pth")
    if os.path.exists(metadata_path):
        try:
            metadata = torch.load(metadata_path)
            print(f"Loaded best model from epoch {metadata['epoch']+1}")
        except Exception as e:
            print(f"Note: Could not load metadata, but model loaded successfully. Error: {e}")
except Exception as e:
    print(f"Error loading model with direct state dict approach: {e}")
    
    # Fall back to legacy loading with weights_only parameter
    try:
        checkpoint = torch.load(best_model_path, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']+1} using legacy method")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded best model state dict using legacy method")
    except Exception as backup_error:
        print(f"Fatal error: Could not load model using any method. Error: {backup_error}")
        print("Continuing with current model state...")

# Evaluation on test set
print("\nEvaluating on test set...")
model.eval()
test_preds = []

# Process test data in batches to handle large datasets efficiently
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
    
    # Concatenate batch predictions
    test_preds = np.concatenate(all_test_preds)
    test_labels = np.concatenate(all_test_labels)
    
    # Calculate all metrics
    test_metrics = calculate_metrics(test_labels, test_preds)
    
    # Print simplified test metrics
    print("\nTest Set Performance:")
    print(f"Correlation: {test_metrics['pearson']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create visualization plot - Only Actual vs Predicted PPG
print("\nCreating PPG prediction visualization...")

# Extract data for plotting
actual_ppg = test_labels[:100, :, 0].reshape(-1)  # Actual PPG from Y_test
predicted_ppg = test_preds[:100, :, 0].reshape(-1)  # Predicted PPG

# Create single plot
plt.figure(figsize=(12, 6))
plt.plot(actual_ppg[:500], label="Actual PPG", color='green', linewidth=2, alpha=0.8)
plt.plot(predicted_ppg[:500], label="Predicted PPG", color='red', linestyle='--', linewidth=2)
plt.title("Actual vs Predicted PPG (First 500 Timepoints)", fontsize=14, fontweight='bold')
plt.xlabel("Time", fontsize=12)
plt.ylabel("Normalized Amplitude", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig(f"ppg_prediction_results_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"PPG prediction plot saved to ppg_prediction_results_{timestamp}.png")
print("\nTraining and evaluation complete!")
