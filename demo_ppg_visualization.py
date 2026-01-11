#!/usr/bin/env python3
"""
PPG Training Pipeline Visualization Demo

This script demonstrates the key components of the PPG training pipeline
with interactive visualizations, including:
- ECG data loading and preprocessing
- PPG template creation and morphology
- PPG signal generation from ECG
- Data sequences for LSTM training
- Model architecture overview

Usage:
    python demo_ppg_visualization.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datetime import datetime
import os

# Import from training modules
from train_improved_ppg_model import (
    load_ecg_data, create_realistic_ppg_template, generate_ppg_from_ecg,
    create_sequences, calculate_metrics
)
from lstm_ppg_model import ECGtoPPG_LSTM

def demo_ecg_loading():
    """Demo ECG data loading"""
    print("=" * 50)
    print("1. ECG DATA LOADING DEMO")
    print("=" * 50)
    
    try:
        ecg = load_ecg_data()
        print(f"✓ Successfully loaded {len(ecg)} ECG samples")
        
        # Show basic statistics
        print(f"ECG Statistics:")
        print(f"  Mean: {np.mean(ecg):.4f}")
        print(f"  Std:  {np.std(ecg):.4f}")
        print(f"  Min:  {np.min(ecg):.4f}")
        print(f"  Max:  {np.max(ecg):.4f}")
        
        return ecg
    except Exception as e:
        print(f"Error loading ECG data: {e}")
        return None

def demo_ppg_template():
    """Demo PPG template creation"""
    print("\n" + "=" * 50)
    print("2. PPG TEMPLATE CREATION DEMO")
    print("=" * 50)
    
    # Create templates with different pulse widths
    pulse_widths = [80, 120, 160]
    templates = []
    
    fig, axes = plt.subplots(1, len(pulse_widths), figsize=(15, 5))
    fig.suptitle('PPG Pulse Templates with Different Widths', fontsize=16)
    
    for i, width in enumerate(pulse_widths):
        template = create_realistic_ppg_template(pulse_width=width)
        templates.append(template)
        
        axes[i].plot(template, 'b-', linewidth=2)
        axes[i].set_title(f'Pulse Width: {width}')
        axes[i].set_xlabel('Sample Index')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True, alpha=0.3)
        
        print(f"✓ Created template with pulse width {width}, length: {len(template)}")
    
    plt.tight_layout()
    plt.show()
    
    return templates[1]  # Return default template

def demo_ppg_generation(ecg_data):
    """Demo PPG generation from ECG"""
    print("\n" + "=" * 50)
    print("3. PPG GENERATION FROM ECG DEMO")
    print("=" * 50)
    
    if ecg_data is None:
        print("No ECG data available for PPG generation")
        return None
    
    # Generate PPG signal
    print("Generating synthetic PPG from ECG...")
    ppg_signal = generate_ppg_from_ecg(ecg_data, delay=120, pulse_width=120)
    
    # Normalize both signals for visualization
    ecg_norm = (ecg_data - ecg_data.mean()) / ecg_data.std()
    ppg_norm = (ppg_signal - ppg_signal.mean()) / ppg_signal.std()
    
    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot first 2000 samples for better visibility
    samples = 2000
    time_axis = np.arange(samples)
    
    # ECG signal
    axes[0].plot(time_axis, ecg_norm[:samples], 'b-', linewidth=1, label='ECG Signal')
    axes[0].set_title('Original ECG Signal', fontsize=14)
    axes[0].set_ylabel('Normalized Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PPG signal
    axes[1].plot(time_axis, ppg_norm[:samples], 'r-', linewidth=1, label='Generated PPG Signal')
    axes[1].set_title('Generated PPG Signal with Realistic Morphology', fontsize=14)
    axes[1].set_ylabel('Normalized Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Both signals overlaid
    axes[2].plot(time_axis, ecg_norm[:samples], 'b-', linewidth=1, alpha=0.7, label='ECG')
    axes[2].plot(time_axis, ppg_norm[:samples], 'r-', linewidth=1, alpha=0.7, label='PPG')
    axes[2].set_title('ECG vs Generated PPG Comparison', fontsize=14)
    axes[2].set_xlabel('Sample Index')
    axes[2].set_ylabel('Normalized Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Generated PPG signal with {len(ppg_signal)} samples")
    print(f"PPG Statistics:")
    print(f"  Mean: {np.mean(ppg_signal):.4f}")
    print(f"  Std:  {np.std(ppg_signal):.4f}")
    
    return ecg_norm, ppg_norm

def demo_sequence_creation(ecg_data, ppg_data):
    """Demo sequence creation for LSTM training"""
    print("\n" + "=" * 50)
    print("4. SEQUENCE CREATION DEMO")
    print("=" * 50)
    
    if ecg_data is None or ppg_data is None:
        print("No data available for sequence creation")
        return None, None
    
    # Create sequences
    seq_len = 100
    X, Y = create_sequences(ecg_data, ppg_data, seq_len=seq_len)
    
    print(f"✓ Created {X.shape[0]} sequences of length {seq_len}")
    print(f"Input sequences shape: {X.shape}")
    print(f"Target sequences shape: {Y.shape}")
    
    # Visualize a few example sequences
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Example Training Sequences (ECG → PPG)', fontsize=16)
    
    for i in range(4):
        row, col = i // 2, i % 2
        seq_idx = i * 200  # Show different sequences
        
        if seq_idx < len(X):
            axes[row, col].plot(X[seq_idx], 'b-', linewidth=1, alpha=0.8, label='ECG Input')
            axes[row, col].plot(Y[seq_idx], 'r-', linewidth=1, alpha=0.8, label='PPG Target')
            axes[row, col].set_title(f'Sequence {seq_idx}')
            axes[row, col].set_xlabel('Time Steps')
            axes[row, col].set_ylabel('Amplitude')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return X, Y

def demo_model_architecture():
    """Demo LSTM model architecture"""
    print("\n" + "=" * 50)
    print("5. LSTM MODEL ARCHITECTURE DEMO")
    print("=" * 50)
    
    # Create model
    model = ECGtoPPG_LSTM()
    
    print("Model Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass with different batch sizes
    batch_sizes = [1, 4, 16]
    seq_len = 100
    
    print("\nForward Pass Tests:")
    model.eval()
    
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, seq_len, 1)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"  Batch size {batch_size}: Input {tuple(test_input.shape)} → Output {tuple(output.shape)}")
    
    return model

def demo_training_simulation(X, Y, model):
    """Demo a short training simulation"""
    print("\n" + "=" * 50)
    print("6. TRAINING SIMULATION DEMO")
    print("=" * 50)
    
    if X is None or Y is None or model is None:
        print("Missing data or model for training simulation")
        return
    
    # Prepare data
    X_tensor = torch.tensor(X[:200, :, np.newaxis]).float()  # Use subset
    Y_tensor = torch.tensor(Y[:200, :, np.newaxis]).float()
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    losses = []
    
    print("Running mini training simulation...")
    
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, Y_tensor)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch % 2 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses)+1), losses, 'b-o', linewidth=2, markersize=6)
    plt.title('Training Loss During Mini Simulation', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"✓ Training simulation complete. Final loss: {losses[-1]:.6f}")
    
    # Test inference
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor[:4])  # Predict on first 4 sequences
    
    # Visualize predictions vs targets
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Predictions vs Targets', fontsize=16)
    
    for i in range(4):
        row, col = i // 2, i % 2
        
        target = Y_tensor[i, :, 0].numpy()
        pred = predictions[i, :, 0].numpy()
        
        axes[row, col].plot(target, 'g-', linewidth=2, label='Target PPG', alpha=0.8)
        axes[row, col].plot(pred, 'r--', linewidth=2, label='Predicted PPG', alpha=0.8)
        axes[row, col].set_title(f'Sequence {i+1}')
        axes[row, col].set_xlabel('Time Steps')
        axes[row, col].set_ylabel('Amplitude')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate metrics
    metrics = calculate_metrics(Y_tensor[:4].numpy(), predictions.numpy())
    print("\nPrediction Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")

def main():
    """Run the complete visualization demo"""
    print("🎯 PPG Training Pipeline Visualization Demo")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run demos in sequence
    ecg_data = demo_ecg_loading()
    template = demo_ppg_template()
    ecg_norm, ppg_norm = demo_ppg_generation(ecg_data)
    X, Y = demo_sequence_creation(ecg_norm, ppg_norm)
    model = demo_model_architecture()
    demo_training_simulation(X, Y, model)
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("All visualizations show the working components of your PPG training pipeline.")
    print("=" * 60)

if __name__ == "__main__":
    main()
