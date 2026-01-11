import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import pickle
import time
import warnings
from scipy.signal import find_peaks, periodogram
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

class AdvancedHRVFeatureExtractor:
    """Enhanced HRV feature extraction with more robust methods"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_comprehensive_features(self, ibi_data, fs=10.0):
        if len(ibi_data) < 10:
            return np.zeros(15)
        
        # Remove outliers using IQR method
        rr_intervals = np.diff(ibi_data) * 1000  # Convert to milliseconds
        q25, q75 = np.percentile(rr_intervals, [25, 75])
        iqr = q75 - q25
        valid_rr = rr_intervals[(rr_intervals > (q25 - 1.5 * iqr)) & 
                               (rr_intervals < (q75 + 1.5 * iqr))]
        
        if len(valid_rr) < 5:
            return np.zeros(15)
        
        # Time domain features
        mean_rr = np.mean(valid_rr)
        sdnn = np.std(valid_rr)
        rmssd = np.sqrt(np.mean(np.diff(valid_rr)**2)) if len(valid_rr) > 1 else 0.0
        nn50 = np.sum(np.abs(np.diff(valid_rr)) > 50)
        pnn50 = nn50 / len(valid_rr) * 100 if len(valid_rr) > 0 else 0.0
        
        # Frequency domain features
        lf_power, hf_power, lf_hf_ratio = self._frequency_features(valid_rr, fs)
        
        # Geometric features
        tri_index = len(valid_rr) / np.max(np.histogram(valid_rr, bins=50)[0]) if len(valid_rr) > 0 else 0
        
        # Nonlinear features
        sample_entropy = self._sample_entropy(valid_rr)
        dfa_alpha1 = self._detrended_fluctuation_analysis(valid_rr)
        
        # Statistical features
        cv = sdnn / mean_rr * 100 if mean_rr > 0 else 0
        mad = np.mean(np.abs(valid_rr - mean_rr))
        
        # Additional robust features
        median_rr = np.median(valid_rr)
        iqr_rr = np.percentile(valid_rr, 75) - np.percentile(valid_rr, 25)
        skewness = self._calculate_skewness(valid_rr)
        kurtosis = self._calculate_kurtosis(valid_rr)
        
        features = np.array([
            mean_rr, sdnn, rmssd, pnn50, lf_power, hf_power, lf_hf_ratio,
            tri_index, sample_entropy, dfa_alpha1, cv, mad, median_rr, iqr_rr, skewness
        ])
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features
    
    def _frequency_features(self, rr_intervals, fs):
        if len(rr_intervals) < 20:
            return 0.0, 0.0, 1.0
            
        t_rr = np.cumsum(np.concatenate([[0], rr_intervals[:-1]]) / 1000.0)
        dt = 1.0 / fs
        t_uniform = np.arange(t_rr[0], t_rr[-1], dt)
        
        if len(t_uniform) < 10:
            return 0.0, 0.0, 1.0
            
        rr_uniform = np.interp(t_uniform, t_rr[:-1], rr_intervals)
        rr_uniform = rr_uniform - np.mean(rr_uniform)
        
        freqs, psd = periodogram(rr_uniform, fs=fs)
        
        lf_band = (freqs >= 0.04) & (freqs <= 0.15)
        hf_band = (freqs >= 0.15) & (freqs <= 0.4)
        
        lf_power = np.sum(psd[lf_band]) if np.any(lf_band) else 0.0
        hf_power = np.sum(psd[hf_band]) if np.any(hf_band) else 0.0
        lf_hf_ratio = lf_power / (hf_power + 1e-8)
        
        return lf_power, hf_power, lf_hf_ratio
    
    def _sample_entropy(self, data, m=2, r=None):
        if r is None:
            r = 0.2 * np.std(data)
        N = len(data)
        if N <= m + 1:
            return 0.0
            
        def _phi(m):
            patterns = np.array([data[i:i+m] for i in range(N-m+1)])
            distances = np.max(np.abs(patterns[:, None] - patterns[None, :]), axis=2)
            matches = np.sum(distances <= r, axis=1) - 1
            return np.sum(matches) / (N-m+1)
        
        phi_m = _phi(m)
        phi_m1 = _phi(m+1)
        
        if phi_m == 0 or phi_m1 == 0:
            return 0.0
        
        return -np.log(phi_m1 / phi_m)
    
    def _detrended_fluctuation_analysis(self, rr_intervals):
        if len(rr_intervals) < 16:
            return 1.0
            
        y = np.cumsum(rr_intervals - np.mean(rr_intervals))
        box_sizes = np.unique(np.logspace(1, np.log10(len(y)//4), 10).astype(int))
        fluctuations = []
        
        for box_size in box_sizes:
            if box_size >= len(y):
                continue
                
            n_boxes = len(y) // box_size
            boxes = y[:n_boxes * box_size].reshape(n_boxes, box_size)
            
            fluctuation = 0
            for box in boxes:
                t = np.arange(len(box))
                coeffs = np.polyfit(t, box, 1)
                trend = np.polyval(coeffs, t)
                fluctuation += np.mean((box - trend)**2)
            
            fluctuations.append(np.sqrt(fluctuation / n_boxes))
        
        if len(fluctuations) < 2:
            return 1.0
            
        log_box_sizes = np.log10(box_sizes[:len(fluctuations)])
        log_fluctuations = np.log10(fluctuations)
        
        if len(log_box_sizes) > 1:
            alpha = np.polyfit(log_box_sizes, log_fluctuations, 1)[0]
            return max(0.5, min(2.0, alpha))
        
        return 1.0
    
    def _calculate_skewness(self, data):
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, data):
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        return np.mean(((data - mean_val) / std_val) ** 4) - 3

class ImprovedEmotionClassifier(nn.Module):
    """Enhanced model architecture with attention mechanism and residual connections"""
    
    def __init__(self, sequence_length=500, d_model=256, nhead=8, 
                 num_transformer_layers=4, num_classes=3, dropout=0.3):
        super(ImprovedEmotionClassifier, self).__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Multi-scale CNN feature extraction
        self.cnn_branch1 = self._create_cnn_branch(1, 32, kernel_sizes=[3, 5, 7])
        self.cnn_branch2 = self._create_cnn_branch(1, 32, kernel_sizes=[9, 11, 13])
        
        # Combine CNN branches
        self.cnn_combine = nn.Conv1d(64, d_model//2, kernel_size=1)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(sequence_length, d_model//2)
        
        # Multi-head self-attention with residual connections
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model//2, nhead=nhead, dim_feedforward=d_model*2,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(15, d_model//4),  # 15 HRV features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//4, d_model//4)
        )
        
        # Final classification layers with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(d_model//2 + d_model//4, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.LayerNorm(d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout//2),
            nn.Linear(d_model//2, num_classes)
        )
        
        self._initialize_weights()
    
    def _create_cnn_branch(self, in_channels, out_channels, kernel_sizes):
        layers = []
        current_channels = in_channels
        
        for i, kernel_size in enumerate(kernel_sizes):
            layers.extend([
                nn.Conv1d(current_channels, out_channels, kernel_size=kernel_size, 
                         padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_channels = out_channels
            
        return nn.Sequential(*layers)
    
    def _create_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x, features):
        batch_size = x.size(0)
        
        # Multi-scale CNN feature extraction
        x_input = x.transpose(1, 2)  # (batch, 1, seq_len)
        
        cnn_out1 = self.cnn_branch1(x_input)
        cnn_out2 = self.cnn_branch2(x_input)
        
        # Combine CNN outputs
        cnn_combined = torch.cat([cnn_out1, cnn_out2], dim=1)
        cnn_out = self.cnn_combine(cnn_combined)
        cnn_out = cnn_out.transpose(1, 2)  # (batch, seq_len, d_model//2)
        
        # Add positional encoding
        pos_enc = self.pos_encoding.to(cnn_out.device)
        cnn_out = cnn_out + pos_enc
        
        # Transformer processing
        transformer_out = self.transformer(cnn_out)
        
        # Global average pooling with attention weights
        attention_weights = torch.softmax(
            torch.mean(transformer_out, dim=-1), dim=-1
        ).unsqueeze(-1)
        sequence_repr = torch.sum(transformer_out * attention_weights, dim=1)
        
        # Process HRV features
        feature_repr = self.feature_fusion(features)
        
        # Combine representations
        combined_repr = torch.cat([sequence_repr, feature_repr], dim=1)
        
        # Final classification
        output = self.classifier(combined_repr)
        
        return output

def create_detailed_confusion_matrix(y_true, y_pred):
    emotion_labels = ['Baseline\n(Calm)', 'Stress\n(Anxious)', 'Amusement\n(Happy)']
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Emotion Classification Results Analysis', fontsize=16, fontweight='bold')
    
    # Raw confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_labels, yticklabels=emotion_labels,
                ax=axes[0,0], cbar_kws={'label': 'Number of Samples'})
    axes[0,0].set_title('Confusion Matrix (Raw Counts)')
    axes[0,0].set_xlabel('Predicted Emotion')
    axes[0,0].set_ylabel('True Emotion')
    
    # Percentage confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=emotion_labels, yticklabels=emotion_labels,
                ax=axes[0,1], cbar_kws={'label': 'Percentage'})
    axes[0,1].set_title('Confusion Matrix (Percentages)')
    axes[0,1].set_xlabel('Predicted Emotion')
    axes[0,1].set_ylabel('True Emotion')
    
    # Per-class accuracy
    class_accuracies = np.diag(cm) / np.sum(cm, axis=1)
    axes[1,0].bar(emotion_labels, class_accuracies, color='lightgreen', alpha=0.7)
    axes[1,0].set_title('Accuracy for Each Emotion')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].set_ylim(0, 1)
    
    # Summary
    axes[1,1].axis('off')
    axes[1,1].text(0.05, 0.95, f"Overall Accuracy: {np.mean(class_accuracies)*100:.1f}%", fontsize=12)
    
    plt.tight_layout()
    plt.show()

def analyze_individual_predictions(model, test_loader, device, num_samples=20):
    model.eval()
    emotion_names = ['Baseline (Calm)', 'Stress (Anxious)', 'Amusement (Happy)']
    
    all_predictions = []
    all_true_labels = []
    all_probabilities = []
    all_signals = []
    
    with torch.no_grad():
        for X_batch, F_batch, y_batch in test_loader:
            X_batch, F_batch = X_batch.to(device), F_batch.to(device)
            outputs = model(X_batch, F_batch)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(y_batch.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_signals.extend(X_batch.cpu().numpy())
            
            if len(all_predictions) >= num_samples:
                break
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Individual Prediction Examples', fontsize=16, fontweight='bold')
    
    for i in range(min(num_samples, len(all_predictions))):
        row = i // 5
        col = i % 5
        
        if row >= 4:
            break
            
        ax = axes[row, col]
        
        signal_data = all_signals[i][:, 0]
        ax.plot(signal_data, linewidth=1.5)
        
        true_emotion = emotion_names[all_true_labels[i]]
        pred_emotion = emotion_names[all_predictions[i]]
        probabilities = all_probabilities[i]
        
        is_correct = all_true_labels[i] == all_predictions[i]
        title_color = 'green' if is_correct else 'red'
        status = '✓ CORRECT' if is_correct else '✗ WRONG'
        
        ax.set_title(f'{status}\nTrue: {true_emotion}\nPred: {pred_emotion}\nConf: {probabilities[all_predictions[i]]:.2f}',
                    fontsize=8, color=title_color, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('IBI (seconds)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def comprehensive_evaluation_with_examples(model, test_loader, device):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    
    with torch.no_grad():
        for X_batch, F_batch, y_batch in test_loader:
            X_batch, F_batch = X_batch.to(device), F_batch.to(device)
            outputs = model(X_batch, F_batch)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probabilities.cpu().numpy())
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_probs = np.array(y_probs)
    
    accuracy = np.mean(y_true == y_pred)
    
    print(f'\nOverall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    
    create_detailed_confusion_matrix(y_true, y_pred)
    analyze_individual_predictions(model, test_loader, device)
    
    print('\nDetailed Performance Metrics:')
    print(classification_report(y_true, y_pred, target_names=['Baseline', 'Stress', 'Amusement'], digits=4))

def load_wesad_data(wesad_dir='WESAD', sequence_length=500, test_size=0.2):
    """Load and preprocess WESAD dataset"""
    print("Loading WESAD dataset...")
    
    all_X, all_F, all_Y = [], [], []
    feature_extractor = AdvancedHRVFeatureExtractor()
    
    # Map labels: 1=baseline, 2=stress, 3=amusement
    label_mapping = {1: 0, 2: 1, 3: 2}  # baseline=0, stress=1, amusement=2
    
    subjects = [f'S{i}' for i in range(2, 18)]  # S2 to S17
    
    for subject in subjects:
        pkl_path = os.path.join(wesad_dir, subject, f'{subject}.pkl')
        
        if not os.path.exists(pkl_path):
            print(f"Warning: {pkl_path} not found, skipping...")
            continue
            
        try:
            print(f"Processing {subject}...")
            
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            # Extract chest sensor data (E4 wristband data)
            if 'signal' in data and 'chest' in data['signal']:
                chest_data = data['signal']['chest']
                labels = data['label']
                
                # Get IBI data (inter-beat intervals)
                if 'IBI' in chest_data:
                    ibi_data = chest_data['IBI']
                    ibi_values = ibi_data[:, 1]  # IBI values in seconds
                    
                    # Create sliding windows
                    for i in range(0, len(ibi_values) - sequence_length, sequence_length // 2):
                        window_ibi = ibi_values[i:i + sequence_length]
                        
                        if len(window_ibi) == sequence_length:
                            # Extract HRV features for this window
                            hrv_features = feature_extractor.extract_comprehensive_features(window_ibi)
                            
                            # Get corresponding label (majority vote for the window)
                            window_labels = labels[i:i + sequence_length]
                            unique_labels, counts = np.unique(window_labels, return_counts=True)
                            majority_label = unique_labels[np.argmax(counts)]
                            
                            # Only use baseline, stress, and amusement labels
                            if majority_label in label_mapping:
                                all_X.append(window_ibi.reshape(-1, 1))
                                all_F.append(hrv_features)
                                all_Y.append(label_mapping[majority_label])
                                
        except Exception as e:
            print(f"Error processing {subject}: {e}")
            continue
    
    if len(all_X) == 0:
        raise ValueError("No valid data found in WESAD dataset")
    
    # Convert to numpy arrays
    X = np.array(all_X)
    F = np.array(all_F)
    Y = np.array(all_Y)
    
    print(f"Loaded {len(X)} samples from WESAD dataset")
    print(f"Data shape: X={X.shape}, F={F.shape}, Y={Y.shape}")
    print(f"Label distribution: {np.bincount(Y)}")
    
    # Split into train and test sets
    X_train, X_test, F_train, F_test, Y_train, Y_test = train_test_split(
        X, F, Y, test_size=test_size, random_state=42, stratify=Y
    )
    
    # Normalize features
    scaler = StandardScaler()
    F_train = scaler.fit_transform(F_train)
    F_test = scaler.transform(F_test)
    
    return (X_train, X_test, F_train, F_test, Y_train, Y_test)

def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=0.001):
    """Train the emotion classification model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, F_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            F_batch = F_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch, F_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, F_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                F_batch = F_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch, F_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        scheduler.step(val_loss)
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 50)
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def main():
    print("WESAD Emotion and Stress Analysis")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load WESAD data
        X_train, X_test, F_train, F_test, Y_train, Y_test = load_wesad_data()
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(F_train),
            torch.LongTensor(Y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(F_test),
            torch.LongTensor(Y_test)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        model = ImprovedEmotionClassifier(
            sequence_length=X_train.shape[1],
            d_model=256,
            nhead=8,
            num_transformer_layers=4,
            num_classes=3,
            dropout=0.3
        ).to(device)
        
        print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train the model
        print("\nStarting training...")
        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, test_loader, device, num_epochs=50
        )
        
        # Evaluate the model
        print("\nEvaluating model on test set...")
        comprehensive_evaluation_with_examples(model, test_loader, device)
        
        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(train_accs, label='Train Accuracy')
        ax2.plot(val_accs, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure the WESAD dataset is properly extracted and available.")
        print("You may need to run the extract_wesad_data.py script first.")

if __name__ == "__main__":
    main()
