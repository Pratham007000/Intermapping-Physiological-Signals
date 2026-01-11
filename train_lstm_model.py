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

# Define device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ECG and PPG data
ecg = np.loadtxt("bidmc01_ecg.csv", delimiter=",")
ppg = np.loadtxt("bidmc01_ppg.csv", delimiter=",")
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

# Lists to store metrics for plotting
train_losses = []
val_losses = []
val_rmse = []
val_r2 = []
val_pearson = []

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
    
    # Store metrics for plotting
    val_rmse.append(val_metrics['rmse'])
    val_r2.append(val_metrics['r2'])
    val_pearson.append(val_metrics['pearson'])
    
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
    
    # Print detailed test metrics
    print("\nTest Set Metrics:")
    print(f"MSE: {test_metrics['mse']:.6f}")
    print(f"RMSE: {test_metrics['rmse']:.6f}")
    print(f"MAE: {test_metrics['mae']:.6f}")
    print(f"R² Score: {test_metrics['r2']:.6f}")
    print(f"Pearson Correlation: {test_metrics['pearson']:.6f}")
    
    # Save test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write(f"Test MSE: {test_metrics['mse']:.6f}\n")
        f.write(f"Test RMSE: {test_metrics['rmse']:.6f}\n")
        f.write(f"Test MAE: {test_metrics['mae']:.6f}\n")
        f.write(f"Test R² Score: {test_metrics['r2']:.6f}\n")
        f.write(f"Test Pearson Correlation: {test_metrics['pearson']:.6f}\n")
        f.write(f"\nModel parameters:\n")
        f.write(f"Hidden size: {model.hidden_size}\n")
        f.write(f"Num layers: {model.num_layers}\n")
        f.write(f"Bidirectional: {model.bidirectional}\n")
        f.write(f"Dropout rate: {model.dropout_rate}\n")
    
    print(f"Test results saved to {results_file}")

# Create visualization plots
print("\nCreating visualization plots...")

# Plot comparison of actual ECG, actual PPG, and predicted PPG
plt.figure(figsize=(15, 10))

# Plot 1: Actual ECG, Actual PPG, and Predicted PPG
plt.subplot(2, 2, 1)
actual_ecg = X_test[:100, :, 0].reshape(-1)  # Extract ECG from input X_test
actual_ppg = test_labels[:100, :, 0].reshape(-1)  # Actual PPG from Y_test
predicted_ppg = test_preds[:100, :, 0].reshape(-1)  # Predicted PPG
plt.plot(actual_ecg[:500], label="Actual ECG", color='blue', alpha=0.7, linewidth=1.5)
plt.plot(actual_ppg[:500], label="Actual PPG", color='green', alpha=0.8)
plt.plot(predicted_ppg[:500], label="Predicted PPG", color='red', linestyle='--')
plt.title("ECG and Actual vs Predicted PPG (First 500 Timepoints)")
plt.xlabel("Time")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Training and Validation Loss
plt.subplot(2, 2, 2)
epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Validation Metrics
plt.subplot(2, 2, 3)
plt.plot(epochs, val_rmse, 'g-', label='RMSE')
plt.plot(epochs, [1-r2 for r2 in val_r2], 'm-', label='1-R²')
plt.title('Validation Metrics Over Time')
plt.xlabel('Epochs')
plt.ylabel('Error Metrics')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Scatter plot of actual vs predicted PPG (first 1000 points)
plt.subplot(2, 2, 4)
plt.scatter(actual_ppg[:1000], predicted_ppg[:1000], alpha=0.5, s=10)
plt.plot([min(actual_ppg[:1000]), max(actual_ppg[:1000])], 
         [min(actual_ppg[:1000]), max(actual_ppg[:1000])], 'r--')
plt.title(f'Predicted vs Actual PPG (r={test_metrics["pearson"]:.4f})')
plt.xlabel('Actual PPG')
plt.ylabel('Predicted PPG')
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save the figure
plt.savefig(f"ppg_prediction_results_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"Results visualization saved to ppg_prediction_results_{timestamp}.png")
print("\nTraining and evaluation complete!")
