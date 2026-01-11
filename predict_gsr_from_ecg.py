import scipy.io
import numpy as np
from scipy import signal
import pywt
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Import the model architecture and preprocessing functions
from ecg_to_gsr_prediction import EnhancedHybridModel, denoise_signal, extract_features

class GSRPredictor:
    def __init__(self, model_path='gsr_model_checkpoints/best_gsr_hybrid_model.pth', window_size=500):
        self.window_size = window_size
        self.fs = 128  # Sampling frequency
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the trained model
        self.model = EnhancedHybridModel(
            input_size=1, 
            d_model=256,
            nhead=8,
            num_transformer_layers=4,
            num_cnn_filters=128,
            dropout=0.6,
            window_size=window_size
        ).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Please train the model first.")
    
    def preprocess_ecg(self, ecg_signal):
        """Preprocess ECG signal for prediction"""
        # Apply denoising
        ecg_denoised = denoise_signal(ecg_signal)
        
        # Robust normalization using quantiles
        ecg_q25, ecg_q75 = np.percentile(ecg_denoised, [25, 75])
        ecg_iqr = ecg_q75 - ecg_q25 + 1e-8
        ecg_normalized = (ecg_denoised - np.median(ecg_denoised)) / ecg_iqr
        ecg_normalized = np.clip(ecg_normalized, -3, 3)
        
        return ecg_normalized
    
    def create_windows(self, ecg_signal):
        """Create overlapping windows from ECG signal"""
        step_size = self.window_size // 2
        num_samples = (len(ecg_signal) - self.window_size) // step_size + 1
        
        if num_samples <= 0:
            raise ValueError(f"ECG signal too short. Need at least {self.window_size} samples, got {len(ecg_signal)}")
        
        X = np.zeros((num_samples, self.window_size, 1))
        F = np.zeros((num_samples, 6))  # 6 features
        
        for i in range(num_samples):
            start = i * step_size
            end = start + self.window_size
            ecg_segment = ecg_signal[start:end]
            X[i, :, 0] = ecg_segment
            F[i] = extract_features(ecg_segment, self.fs)
        
        return torch.FloatTensor(X).to(self.device), torch.FloatTensor(F).to(self.device)
    
    def predict_gsr(self, ecg_signal):
        """Predict GSR from ECG signal"""
        # Preprocess the ECG signal
        ecg_preprocessed = self.preprocess_ecg(ecg_signal)
        
        # Create windows
        X, F = self.create_windows(ecg_preprocessed)
        
        # Perform prediction
        with torch.no_grad():
            predicted_gsr_windows = self.model(X, F)
        
        # Convert back to numpy
        predicted_gsr_windows = predicted_gsr_windows.cpu().numpy()
        
        # Reconstruct full GSR signal from overlapping windows
        step_size = self.window_size // 2
        full_length = len(ecg_preprocessed)
        predicted_gsr_full = np.zeros(full_length)
        count_matrix = np.zeros(full_length)
        
        for i, window_gsr in enumerate(predicted_gsr_windows):
            start = i * step_size
            end = start + self.window_size
            if end <= full_length:
                predicted_gsr_full[start:end] += window_gsr
                count_matrix[start:end] += 1
        
        # Average overlapping regions
        predicted_gsr_full = np.divide(predicted_gsr_full, count_matrix, 
                                     out=np.zeros_like(predicted_gsr_full), 
                                     where=count_matrix!=0)
        
        return predicted_gsr_full, predicted_gsr_windows

def load_ecg_data(file_path):
    """Load ECG data from .mat file"""
    try:
        data = scipy.io.loadmat(file_path)
        
        # Find ECG data key
        ecg_keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"Available keys in {os.path.basename(file_path)}: {ecg_keys}")
        
        ecg = None
        for key in ['ECGdata', 'ecg', 'ECG', 'data']:
            if key in data:
                ecg = data[key]
                break
        
        if ecg is None and ecg_keys:
            ecg = data[ecg_keys[0]]
        
        if isinstance(ecg, np.ndarray) and ecg.ndim > 1:
            ecg = ecg.flatten()
        
        ecg = np.array(ecg).flatten()
        return ecg
        
    except Exception as e:
        print(f"Error loading ECG data from {file_path}: {e}")
        return None

def load_gsr_data(file_path):
    """Load GSR data from .mat file for comparison"""
    try:
        data = scipy.io.loadmat(file_path)
        
        # Find GSR data key
        gsr_keys = [k for k in data.keys() if not k.startswith('__')]
        
        gsr = None
        for key in ['GSRdata', 'gsr', 'GSR', 'data']:
            if key in data:
                gsr = data[key]
                break
        
        if gsr is None and gsr_keys:
            gsr = data[gsr_keys[0]]
        
        if isinstance(gsr, np.ndarray) and gsr.ndim > 1:
            gsr = gsr.flatten()
        
        gsr = np.array(gsr).flatten()
        return gsr
        
    except Exception as e:
        print(f"Error loading GSR data from {file_path}: {e}")
        return None

def visualize_prediction(ecg_signal, predicted_gsr, actual_gsr=None, subject_id="Unknown"):
    """Visualize ECG signal and predicted GSR"""
    time_axis = np.arange(len(ecg_signal)) / 128  # Convert to seconds
    
    plt.figure(figsize=(15, 10))
    
    # Plot ECG signal
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, ecg_signal, 'b-', linewidth=1)
    plt.title(f'ECG Signal - Subject {subject_id}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('ECG Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Plot predicted GSR
    plt.subplot(3, 1, 2)
    gsr_time_axis = np.arange(len(predicted_gsr)) / 128
    plt.plot(gsr_time_axis, predicted_gsr, 'r-', linewidth=2, label='Predicted GSR')
    
    if actual_gsr is not None:
        actual_time_axis = np.arange(len(actual_gsr)) / 128
        plt.plot(actual_time_axis, actual_gsr, 'g-', linewidth=2, label='Actual GSR', alpha=0.7)
        plt.legend()
        
        # Calculate and display metrics
        min_len = min(len(predicted_gsr), len(actual_gsr))
        mae = np.mean(np.abs(predicted_gsr[:min_len] - actual_gsr[:min_len]))
        plt.title(f'GSR Prediction vs Actual - Subject {subject_id} (MAE: {mae:.4f})')
    else:
        plt.title(f'Predicted GSR - Subject {subject_id}')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('GSR Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Plot correlation if actual GSR is available
    if actual_gsr is not None:
        plt.subplot(3, 1, 3)
        min_len = min(len(predicted_gsr), len(actual_gsr))
        plt.scatter(actual_gsr[:min_len], predicted_gsr[:min_len], alpha=0.6)
        plt.plot([actual_gsr[:min_len].min(), actual_gsr[:min_len].max()], 
                [actual_gsr[:min_len].min(), actual_gsr[:min_len].max()], 'r--', lw=2)
        plt.title('Predicted vs Actual GSR Correlation')
        plt.xlabel('Actual GSR')
        plt.ylabel('Predicted GSR')
        plt.grid(True, alpha=0.3)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(predicted_gsr[:min_len], actual_gsr[:min_len])[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def main():
    # Initialize the predictor
    predictor = GSRPredictor()
    
    # Path to the dataset
    base_dir = os.path.expanduser('~/Desktop/PPG_Estimation_Project/g2p7vwxyn2-1/ECG_GSR_Emotions/Raw Data/Multimodal')
    ecg_dir = os.path.join(base_dir, 'ECG')
    gsr_dir = os.path.join(base_dir, 'GSR')
    
    # Find all ECG files
    ecg_files = glob.glob(os.path.join(ecg_dir, 'ECGdata_*.mat'))
    
    if not ecg_files:
        print(f"No ECG files found in {ecg_dir}")
        return
    
    print(f"Found {len(ecg_files)} ECG files")
    
    # Process each file
    for ecg_file in ecg_files[:3]:  # Process first 3 files for demonstration
        # Extract subject ID
        filename = os.path.basename(ecg_file)
        subject_id = filename.replace('ECGdata_', '').replace('.mat', '')
        
        print(f"\n{'='*50}")
        print(f"Processing Subject: {subject_id}")
        print(f"ECG File: {filename}")
        
        # Load ECG data
        ecg_signal = load_ecg_data(ecg_file)
        if ecg_signal is None or len(ecg_signal) < 500:
            print(f"Skipping {subject_id}: Invalid or too short ECG data")
            continue
        
        # Load corresponding GSR data if available
        gsr_file = os.path.join(gsr_dir, f'GSRdata_{subject_id}.mat')
        actual_gsr = None
        if os.path.exists(gsr_file):
            actual_gsr = load_gsr_data(gsr_file)
            print(f"GSR File: GSRdata_{subject_id}.mat")
        else:
            print(f"No corresponding GSR file found for {subject_id}")
        
        try:
            # Predict GSR from ECG
            print(f"ECG signal length: {len(ecg_signal)} samples ({len(ecg_signal)/128:.1f} seconds)")
            
            predicted_gsr_full, predicted_gsr_windows = predictor.predict_gsr(ecg_signal)
            
            print(f"Predicted GSR length: {len(predicted_gsr_full)} samples")
            print(f"Number of prediction windows: {len(predicted_gsr_windows)}")
            print(f"Predicted GSR range: [{predicted_gsr_full.min():.3f}, {predicted_gsr_full.max():.3f}]")
            print(f"Predicted GSR mean: {predicted_gsr_full.mean():.3f} ± {predicted_gsr_full.std():.3f}")
            
            # Compare with actual GSR if available
            if actual_gsr is not None:
                min_len = min(len(predicted_gsr_full), len(actual_gsr))
                mae = np.mean(np.abs(predicted_gsr_full[:min_len] - actual_gsr[:min_len]))
                correlation = np.corrcoef(predicted_gsr_full[:min_len], actual_gsr[:min_len])[0, 1]
                print(f"MAE vs Actual GSR: {mae:.4f}")
                print(f"Correlation with Actual GSR: {correlation:.3f}")
            
            # Visualize results
            visualize_prediction(ecg_signal, predicted_gsr_full, actual_gsr, subject_id)
            
            # Save predictions
            output_dir = 'gsr_predictions'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as numpy arrays
            np.save(os.path.join(output_dir, f'predicted_gsr_{subject_id}.npy'), predicted_gsr_full)
            np.save(os.path.join(output_dir, f'ecg_signal_{subject_id}.npy'), ecg_signal)
            
            print(f"Predictions saved to {output_dir}/predicted_gsr_{subject_id}.npy")
            
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
    
    print(f"\n{'='*50}")
    print("GSR prediction completed!")

if __name__ == "__main__":
    main()
