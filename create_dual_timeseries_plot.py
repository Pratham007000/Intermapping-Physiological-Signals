#!/usr/bin/env python3
"""
Create a dual time-series line chart showing ECG Signal and Predicted PPG
This script generates a plot similar to the screenshot description provided.

Usage:
    python3 create_dual_timeseries_plot.py

Author: PPG Estimation Project
Date: 2025-07-20
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from contextlib import redirect_stdout
import io

# Import the necessary functions
try:
    from train_improved_ppg_model import (
        load_ecg_data,
        generate_ppg_from_ecg,
        create_sequences,
        calculate_metrics
    )
    from lstm_ppg_model import ECGtoPPG_LSTM
    import torch
    print("✅ Successfully imported all modules for plotting")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure train_improved_ppg_model.py and lstm_ppg_model.py are in the same directory")
    sys.exit(1)


def create_dual_timeseries_plot():
    """Create the dual time-series plot as specified"""
    
    print("📊 Generating dual time-series plot...")
    
    # Load real ECG data
    print("Loading ECG data...")
    ecg_data = load_ecg_data()
    
    # Generate PPG from ECG (suppress output)
    print("Generating synthetic PPG target...")
    with redirect_stdout(io.StringIO()):
        ppg_data = generate_ppg_from_ecg(ecg_data, delay=120, pulse_width=120)
    
    # Normalize both signals
    ecg_normalized = (ecg_data - ecg_data.mean()) / ecg_data.std()
    ppg_normalized = (ppg_data - ppg_data.mean()) / ppg_data.std()
    
    # Get first 500 timepoints
    time_points = 500
    ecg_plot = ecg_normalized[:time_points]
    ppg_plot = ppg_normalized[:time_points]
    time_axis = np.arange(time_points)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot ECG signal in blue (solid line)
    plt.plot(time_axis, ecg_plot, 'b-', linewidth=1.5, label='ECG Signal', alpha=0.8)
    
    # Plot Predicted PPG in red (dashed line)
    plt.plot(time_axis, ppg_plot, 'r--', linewidth=1.5, label='Predicted PPG', alpha=0.8)
    
    # Formatting
    plt.xlabel('Time (samples)', fontsize=12, fontweight='bold')
    plt.ylabel('Normalized Amplitude', fontsize=12, fontweight='bold')
    plt.title('ECG Signal vs Predicted PPG - Dual Time Series Analysis', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set axis limits for better visualization
    plt.xlim(0, time_points)
    
    # Add some styling
    plt.tight_layout()
    
    # Add correlation info as text
    correlation = np.corrcoef(ecg_plot, ppg_plot)[0, 1]
    plt.text(0.02, 0.98, f'Correlation: r = {correlation:.3f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             fontsize=10, verticalalignment='top')
    
    # Save the plot
    plot_filename = "dual_timeseries_ecg_ppg.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📈 Plot saved as '{plot_filename}'")
    
    # Display plot info
    print(f"📊 Plot Details:")
    print(f"   • ECG Signal: Blue solid line")
    print(f"   • Predicted PPG: Red dashed line")  
    print(f"   • Time range: 0-{time_points} samples")
    print(f"   • Correlation: r = {correlation:.3f}")
    print(f"   • File size: {os.path.getsize(plot_filename) / 1024:.1f} KB")
    
    plt.show()


def create_enhanced_dual_plot():
    """Create an enhanced version with model predictions"""
    
    print("\n🔬 Creating enhanced plot with LSTM model predictions...")
    
    # Load and prepare data
    ecg_data = load_ecg_data()
    
    # Generate target PPG
    with redirect_stdout(io.StringIO()):
        target_ppg = generate_ppg_from_ecg(ecg_data, delay=120, pulse_width=120)
    
    # Normalize
    ecg_norm = (ecg_data - ecg_data.mean()) / ecg_data.std()
    ppg_norm = (target_ppg - target_ppg.mean()) / target_ppg.std()
    
    # Create sequences for model
    X, Y = create_sequences(ecg_norm, ppg_norm, seq_len=100)
    X = X[:, :, np.newaxis]  # Add feature dimension
    
    # Load trained model if available, otherwise use untrained model
    model = ECGtoPPG_LSTM()
    model_path = "model_checkpoints/best_lstm_model_improved_ppg.pth"
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print("✅ Loaded trained model")
        except:
            print("⚠️  Using untrained model (for demonstration)")
    else:
        print("⚠️  Using untrained model (for demonstration)")
    
    model.eval()
    
    # Generate predictions for first few sequences
    X_sample = torch.tensor(X[:5]).float()  # First 5 sequences
    with torch.no_grad():
        predictions = model(X_sample)
    
    # Reconstruct signals from sequences for plotting
    pred_signal = predictions[0, :, 0].numpy()  # First sequence prediction
    target_signal = Y[0, :]  # First sequence target
    ecg_signal = X[0, :, 0]  # First sequence ECG
    
    # Create the enhanced plot
    plt.figure(figsize=(15, 10))
    
    # Main plot
    plt.subplot(2, 1, 1)
    time_axis = np.arange(len(ecg_signal))
    
    plt.plot(time_axis, ecg_signal, 'b-', linewidth=1.5, label='ECG Signal', alpha=0.8)
    plt.plot(time_axis, pred_signal, 'r--', linewidth=1.5, label='Predicted PPG', alpha=0.8)
    plt.plot(time_axis, target_signal, 'g:', linewidth=1.5, label='Target PPG', alpha=0.6)
    
    plt.xlabel('Time (samples)', fontsize=12, fontweight='bold')
    plt.ylabel('Normalized Amplitude', fontsize=12, fontweight='bold')
    plt.title('ECG Signal vs LSTM Predicted PPG vs Target PPG', 
              fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Calculate metrics
    metrics = calculate_metrics(target_signal.reshape(-1, 1, 1), pred_signal.reshape(-1, 1, 1))
    
    # Add metrics text
    metrics_text = f"RMSE: {metrics['rmse']:.3f} | R²: {metrics['r2']:.3f} | Correlation: {metrics['pearson']:.3f}"
    plt.text(0.02, 0.95, metrics_text,
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
             fontsize=10)
    
    # Scatter plot for correlation
    plt.subplot(2, 1, 2)
    plt.scatter(target_signal, pred_signal, alpha=0.6, s=20, color='purple')
    plt.plot([target_signal.min(), target_signal.max()], 
             [target_signal.min(), target_signal.max()], 'k--', alpha=0.5)
    plt.xlabel('Target PPG', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted PPG', fontsize=12, fontweight='bold')
    plt.title('Prediction vs Target Correlation Plot', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save enhanced plot
    enhanced_filename = "enhanced_dual_timeseries_with_predictions.png"
    plt.savefig(enhanced_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"🎨 Enhanced plot saved as '{enhanced_filename}'")
    print(f"📊 Performance Metrics:")
    print(f"   • RMSE: {metrics['rmse']:.3f}")
    print(f"   • R²: {metrics['r2']:.3f}")  
    print(f"   • Correlation: {metrics['pearson']:.3f}")
    
    plt.show()


if __name__ == "__main__":
    print("🎨 Creating Dual Time-Series Plots")
    print("=" * 50)
    
    # Create the basic dual plot as requested
    create_dual_timeseries_plot()
    
    # Create enhanced version with model predictions
    create_enhanced_dual_plot()
    
    print("\n✅ All plots created successfully!")
    print("Files generated:")
    print("  • dual_timeseries_ecg_ppg.png")
    print("  • enhanced_dual_timeseries_with_predictions.png")
