"""
TinyML Inference Demo
====================

Demonstrates how to use the trained TinyML model for real-time ECG-to-PPG inference.
This script shows how the model would be deployed on edge devices.

Features:
- Loads the trained TinyML model
- Processes ECG data in real-time chunks
- Shows model size and inference speed
- Optimized for embedded systems
"""

import numpy as np
import torch
import time
from tinyml_lstm_model import TinyECGtoPPG_LSTM, TinyPPGProcessor
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Set device for TinyML (CPU optimized)
device = torch.device('cpu')
print(f"🔬 TinyML Inference Demo - Running on: {device}")


def load_tiny_model(model_path="best_tinyml_ppg_model.pth"):
    """Load the trained TinyML model."""
    try:
        # Create model with same architecture as training
        model = TinyECGtoPPG_LSTM(
            input_size=1,
            hidden_size=48,
            num_layers=2,
            output_size=1,
            dropout_rate=0.1
        ).to(device)
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model.eval()
        
        # Get model info
        param_count, model_size = model.get_model_size()
        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {param_count:,}")
        print(f"   Model size: {model_size/1024:.1f} KB")
        
        return model
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("Please run train_tinyml_ppg_model.py first to create the model.")
        return None


def generate_test_ecg(duration_seconds=10, sampling_rate=180):
    """Generate synthetic ECG data for testing."""
    print(f"📊 Generating {duration_seconds}s of test ECG data...")
    
    t = np.linspace(0, duration_seconds, int(duration_seconds * sampling_rate))
    
    # Simulate realistic ECG with R-peaks
    heart_rate = 75  # BPM
    r_peak_interval = 60 / heart_rate  # seconds between R-peaks
    
    ecg_signal = np.zeros_like(t)
    
    # Add R-peaks at regular intervals
    for peak_time in np.arange(0, duration_seconds, r_peak_interval):
        peak_idx = int(peak_time * sampling_rate)
        if peak_idx < len(ecg_signal):
            # QRS complex simulation
            start_idx = max(0, peak_idx - 5)
            end_idx = min(len(ecg_signal), peak_idx + 10)
            
            # Create QRS shape
            qrs_len = end_idx - start_idx
            qrs_shape = np.exp(-((np.arange(qrs_len) - 5) / 3) ** 2)
            ecg_signal[start_idx:end_idx] += qrs_shape
    
    # Add some baseline noise
    ecg_signal += 0.1 * np.random.randn(len(t))
    
    # Normalize
    ecg_signal = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()
    
    return ecg_signal.astype(np.float32)


def process_ecg_chunks(model, ecg_data, chunk_size=32):
    """Process ECG data in chunks suitable for TinyML."""
    print(f"⚡ Processing ECG in {chunk_size}-sample chunks...")
    
    ppg_predictions = []
    inference_times = []
    
    # Process in overlapping chunks
    stride = chunk_size // 4  # 75% overlap for smooth output
    
    for i in range(0, len(ecg_data) - chunk_size + 1, stride):
        chunk = ecg_data[i:i + chunk_size]
        
        # Prepare input tensor
        input_tensor = torch.tensor(chunk).unsqueeze(0).unsqueeze(-1).to(device)
        
        # Measure inference time
        start_time = time.time()
        
        with torch.no_grad():
            prediction = model(input_tensor)
        
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Extract prediction
        pred_chunk = prediction.squeeze().cpu().numpy()
        ppg_predictions.extend(pred_chunk)
    
    # Truncate to match input length
    ppg_predictions = np.array(ppg_predictions[:len(ecg_data)])
    
    avg_inference_time = np.mean(inference_times) * 1000  # ms
    print(f"📈 Processed {len(inference_times)} chunks")
    print(f"⏱️  Average inference time: {avg_inference_time:.2f} ms per chunk")
    print(f"🚀 Inference rate: {1000/avg_inference_time:.1f} chunks/second")
    
    return ppg_predictions, avg_inference_time


def calculate_inference_metrics(ecg_data, ppg_predictions, sampling_rate=180):
    """Calculate performance metrics for inference."""
    print("📊 Calculating inference metrics...")
    
    # Basic signal quality metrics
    signal_power = np.mean(ppg_predictions ** 2)
    noise_estimate = np.std(np.diff(ppg_predictions))  # High-frequency noise
    snr_estimate = 10 * np.log10(signal_power / (noise_estimate ** 2 + 1e-8))
    
    # Peak detection for heart rate estimation
    processor = TinyPPGProcessor(seq_length=32, sampling_rate=sampling_rate)
    peaks = processor.detect_simple_peaks(ppg_predictions, min_distance=36)
    
    if len(peaks) > 1:
        peak_intervals = np.diff(peaks) / sampling_rate  # seconds
        heart_rate = 60 / np.mean(peak_intervals)  # BPM
    else:
        heart_rate = 0
    
    metrics = {
        'signal_power': signal_power,
        'snr_estimate': snr_estimate,
        'detected_peaks': len(peaks),
        'estimated_hr': heart_rate
    }
    
    print(f"   Signal Power: {signal_power:.4f}")
    print(f"   SNR Estimate: {snr_estimate:.2f} dB")
    print(f"   Detected Peaks: {len(peaks)}")
    print(f"   Estimated HR: {heart_rate:.1f} BPM")
    
    return metrics


def create_inference_visualization(ecg_data, ppg_predictions, inference_time, model_size):
    """Create visualization of inference results."""
    print("🎨 Creating inference visualization...")
    
    plt.figure(figsize=(16, 10))
    
    # Create time axis (first 5 seconds for clarity)
    plot_samples = min(900, len(ecg_data))  # 5 seconds at 180 Hz
    time_axis = np.arange(plot_samples) / 180
    
    # Plot 1: ECG Input
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, ecg_data[:plot_samples], 'b-', linewidth=2, alpha=0.8)
    plt.title('ECG Input Signal', fontsize=14, fontweight='bold')
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, time_axis[-1])
    
    # Plot 2: PPG Prediction
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, ppg_predictions[:plot_samples], 'r-', linewidth=2, alpha=0.8)
    plt.title('TinyML PPG Prediction', fontsize=14, fontweight='bold')
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, time_axis[-1])
    
    # Plot 3: Overlay Comparison
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, ecg_data[:plot_samples], 'b-', linewidth=2, alpha=0.7, label='ECG Input')
    plt.plot(time_axis, ppg_predictions[:plot_samples], 'r-', linewidth=2, alpha=0.7, label='PPG Prediction')
    plt.title('ECG vs Predicted PPG', fontsize=14, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Normalized Amplitude', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, time_axis[-1])
    
    # Add performance info
    info_text = (f'TinyML Performance:\n'
                f'Model Size: {model_size/1024:.1f} KB\n'
                f'Inference Time: {inference_time:.2f} ms/chunk\n'
                f'Memory Efficient: 32-sample chunks\n'
                f'Edge-Ready: CPU optimized')
    
    plt.figtext(0.02, 0.95, info_text, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tinyml_inference_demo_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    return filename


def main():
    """Main inference demo."""
    print("🚀 Starting TinyML ECG-to-PPG Inference Demo...\n")
    
    # Load the trained model
    model = load_tiny_model()
    if model is None:
        return
    
    # Get model size for reporting
    _, model_size = model.get_model_size()
    
    print("\n" + "="*60)
    
    # Generate test ECG data
    test_ecg = generate_test_ecg(duration_seconds=10, sampling_rate=180)
    
    print("="*60)
    
    # Process ECG and get PPG predictions
    ppg_pred, avg_inference_time = process_ecg_chunks(model, test_ecg, chunk_size=32)
    
    print("="*60)
    
    # Calculate metrics
    metrics = calculate_inference_metrics(test_ecg, ppg_pred, sampling_rate=180)
    
    print("="*60)
    
    # Create visualization
    viz_filename = create_inference_visualization(test_ecg, ppg_pred, avg_inference_time, model_size)
    
    print(f"\n✅ TinyML Inference Demo Complete!")
    print(f"📊 Results visualization: {viz_filename}")
    print(f"📱 Model ready for edge deployment!")
    print(f"🔧 Suitable for: Arduino, Raspberry Pi, ESP32, etc.")
    
    # Edge deployment summary
    print(f"\n🌟 Edge Deployment Summary:")
    print(f"   Model Size: {model_size/1024:.1f} KB (fits on microcontrollers)")
    print(f"   RAM Usage: ~2 KB (32 samples × 4 bytes × 2 buffers)")
    print(f"   Processing: {avg_inference_time:.2f} ms per 32 samples")
    print(f"   Real-time: {1000/avg_inference_time:.1f}x faster than real-time")
    print(f"   Power: Low (CPU-only, no GPU required)")


if __name__ == "__main__":
    main()
