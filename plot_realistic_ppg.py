#!/usr/bin/env python3
"""
Realistic PPG Data Plotter for Arduino Results
==============================================
This script creates professional PPG visualizations from real Arduino data.
Supports multiple data formats and provides clinical-grade visualization.

Usage:
python3 plot_realistic_ppg.py [--data PATH] [--mode MODE] [--save FILENAME]
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys
from datetime import datetime
import serial
import threading
import queue
import time
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

def load_realistic_ppg_data():
    """Load the realistic ECG-PPG data from CSV"""
    try:
        df = pd.read_csv('realistic_ecg_ppg_data.csv')
        print(f"Loaded {len(df)} samples from realistic_ecg_ppg_data.csv")
        return df
    except FileNotFoundError:
        print("Warning: realistic_ecg_ppg_data.csv not found!")
        return None

def load_bidmc_data():
    """Load BIDMC dataset if available"""
    try:
        ecg_df = pd.read_csv('bidmc01_ecg.csv')
        ppg_df = pd.read_csv('bidmc01_ppg.csv')
        print(f"Loaded BIDMC data: {len(ecg_df)} ECG samples, {len(ppg_df)} PPG samples")
        
        # Align the data
        min_len = min(len(ecg_df), len(ppg_df))
        combined_df = pd.DataFrame({
            'ECG Amplitude': ecg_df.iloc[:min_len, 0],  # First column
            'PPG Amplitude': ppg_df.iloc[:min_len, 0]   # First column
        })
        return combined_df
    except FileNotFoundError:
        print("Warning: BIDMC data files not found!")
        return None

def parse_arduino_serial_data(data_string):
    """Parse Arduino serial output data"""
    lines = data_string.strip().split('\n')
    data = []
    
    for line in lines:
        if line.strip() and ',' in line:
            parts = line.split(',')
            try:
                if len(parts) >= 5:  # timestamp,sample_index,ecg,ppg,heart_rate format
                    timestamp = int(parts[0])
                    sample_index = int(parts[1])
                    ecg = float(parts[2])
                    ppg = float(parts[3])
                    heart_rate = float(parts[4])
                    
                    data.append({
                        'timestamp': timestamp,
                        'sample_index': sample_index,
                        'ECG Amplitude': ecg,
                        'PPG Amplitude': ppg,
                        'heart_rate': heart_rate
                    })
            except (ValueError, IndexError):
                continue
    
    return pd.DataFrame(data) if data else pd.DataFrame()

def filter_signal(signal_data, fs=360, lowcut=0.5, highcut=8.0):
    """Apply bandpass filter to PPG signal"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if high >= 1.0:
        high = 0.99
    
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal_data)
    return filtered_signal

def detect_ppg_peaks(ppg_data, fs=360):
    """Detect peaks in PPG signal for heart rate calculation"""
    # Filter the signal first
    filtered_ppg = filter_signal(ppg_data, fs)
    
    # Find peaks with appropriate parameters for PPG
    min_distance = int(fs * 0.6)  # Minimum 0.6 seconds between peaks (100 BPM max)
    peaks, properties = find_peaks(
        filtered_ppg, 
        distance=min_distance,
        prominence=np.std(filtered_ppg) * 0.3,
        height=np.mean(filtered_ppg)
    )
    
    return peaks, filtered_ppg

def calculate_heart_rate_from_peaks(peaks, fs=360, window_size=10):
    """Calculate heart rate from detected peaks"""
    if len(peaks) < 2:
        return np.array([60.0])  # Default heart rate
    
    # Calculate intervals between peaks (in seconds)
    intervals = np.diff(peaks) / fs
    
    # Convert to heart rates (beats per minute)
    heart_rates = 60.0 / intervals
    
    # Apply moving average smoothing
    if len(heart_rates) >= window_size:
        kernel = np.ones(window_size) / window_size
        heart_rates = np.convolve(heart_rates, kernel, mode='valid')
    
    # Clip to reasonable physiological range
    heart_rates = np.clip(heart_rates, 40, 200)
    
    return heart_rates

def create_professional_ppg_plot(df, title="Professional PPG Analysis", save_path=None):
    """Create a professional medical-grade PPG visualization"""
    
    # Set up the plot style
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 12))
    
    # Create time axis
    if 'timestamp' in df.columns:
        time_axis = (df['timestamp'] - df['timestamp'].iloc[0])
        xlabel = 'Time (seconds)'
    else:
        time_axis = np.arange(len(df)) / 360.0  # Assume 360 Hz sampling rate
        xlabel = 'Time (seconds)'
    
    # Convert to numpy array for safer indexing
    time_axis = np.array(time_axis)
    
    # Extract signals - handle different column naming conventions
    if 'ECG Amplitude' in df.columns:
        ecg_signal = df['ECG Amplitude'].values
        ppg_signal = df['PPG Amplitude'].values
    elif 'ecg' in df.columns:
        ecg_signal = df['ecg'].values
        ppg_signal = df['ppg'].values
    else:
        # Try to find ECG/PPG columns by position
        cols = list(df.columns)
        ecg_signal = df.iloc[:, 1].values if len(cols) > 1 else np.zeros(len(df))
        ppg_signal = df.iloc[:, 2].values if len(cols) > 2 else np.zeros(len(df))
    
    # Filter signals for better visualization
    ppg_filtered = filter_signal(ppg_signal)
    ecg_filtered = filter_signal(ecg_signal, lowcut=1.0, highcut=25.0)
    
    # Detect PPG peaks
    ppg_peaks, _ = detect_ppg_peaks(ppg_signal)
    
    # Calculate heart rate
    hr_values = calculate_heart_rate_from_peaks(ppg_peaks)
    
    # Create subplots
    gs = fig.add_gridspec(4, 2, height_ratios=[2, 2, 1.5, 1], hspace=0.3, wspace=0.25)
    
    # 1. Raw PPG Signal
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_axis, ppg_signal, color='#E74C3C', linewidth=1, alpha=0.7, label='Raw PPG')
    ax1.plot(time_axis, ppg_filtered, color='#C0392B', linewidth=1.5, label='Filtered PPG')
    
    # Mark detected peaks
    if len(ppg_peaks) > 0:
        ax1.plot(time_axis[ppg_peaks], ppg_signal[ppg_peaks], 'ro', markersize=4, alpha=0.8, label='Detected Peaks')
    
    ax1.set_title('Photoplethysmogram (PPG) Signal Analysis', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('PPG Amplitude', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', framealpha=0.9)
    
    # Add signal statistics
    ppg_stats = f'Mean: {np.mean(ppg_signal):.3f}, Std: {np.std(ppg_signal):.3f}, SNR: {np.mean(ppg_signal)/np.std(ppg_signal):.2f}'
    ax1.text(0.02, 0.95, ppg_stats, transform=ax1.transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # 2. ECG Signal (for comparison)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(time_axis, ecg_signal, color='#27AE60', linewidth=1, alpha=0.7, label='Raw ECG')
    ax2.plot(time_axis, ecg_filtered, color='#196F3D', linewidth=1.5, label='Filtered ECG')
    ax2.set_title('Electrocardiogram (ECG) Reference Signal', fontsize=14, fontweight='bold')
    ax2.set_ylabel('ECG Amplitude', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', framealpha=0.9)
    
    # 3. Heart Rate Analysis (left subplot)
    ax3 = fig.add_subplot(gs[2, 0])
    if len(hr_values) > 0:
        hr_time = time_axis[ppg_peaks[1:len(hr_values)+1]]  # Align with heart rate values
        ax3.plot(hr_time, hr_values, color='#8E44AD', linewidth=2, marker='o', markersize=3, label='Instantaneous HR')
        ax3.axhline(y=np.mean(hr_values), color='red', linestyle='--', alpha=0.7, label=f'Mean HR: {np.mean(hr_values):.1f} BPM')
        ax3.fill_between(hr_time, np.mean(hr_values) - np.std(hr_values), np.mean(hr_values) + np.std(hr_values), 
                        alpha=0.2, color='red', label='±1σ')
    else:
        ax3.text(0.5, 0.5, 'No peaks detected\nfor HR calculation', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12, color='red')
    
    ax3.set_title('Heart Rate Analysis', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (seconds)', fontsize=10)
    ax3.set_ylabel('Heart Rate (BPM)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_ylim(40, 120)
    
    # 4. Frequency Analysis (right subplot)
    ax4 = fig.add_subplot(gs[2, 1])
    
    # Compute power spectral density
    frequencies, psd = signal.welch(ppg_filtered, fs=360, nperseg=min(1024, len(ppg_filtered)//4))
    
    # Focus on physiological range (0.5-4 Hz corresponds to 30-240 BPM)
    freq_mask = (frequencies >= 0.5) & (frequencies <= 4.0)
    frequencies = frequencies[freq_mask]
    psd = psd[freq_mask]
    
    ax4.semilogy(frequencies * 60, psd, color='#F39C12', linewidth=2)  # Convert to BPM
    ax4.set_title('PPG Frequency Spectrum', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Frequency (BPM)', fontsize=10)
    ax4.set_ylabel('Power Spectral Density', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Mark dominant frequency
    if len(psd) > 0:
        peak_freq_idx = np.argmax(psd)
        dominant_freq = frequencies[peak_freq_idx] * 60  # Convert to BPM
        ax4.axvline(x=dominant_freq, color='red', linestyle='--', alpha=0.7, 
                   label=f'Dominant: {dominant_freq:.1f} BPM')
        ax4.legend(fontsize=9)
    
    # 5. Signal Quality Metrics (bottom)
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    
    # Calculate quality metrics
    duration = time_axis[-1] - time_axis[0]
    samples = len(df)
    sampling_rate = samples / duration if duration > 0 else 360
    
    # Peak detection quality
    peak_quality = len(ppg_peaks) / (duration / 60) if duration > 0 else 0  # peaks per minute
    
    # Signal quality metrics
    snr_ppg = np.mean(ppg_signal) / np.std(ppg_signal) if np.std(ppg_signal) > 0 else 0
    
    metrics_text = f"""
    📊 SIGNAL QUALITY ANALYSIS
    
    ⏱️  Duration: {duration:.1f} seconds ({samples:,} samples)
    📡 Sampling Rate: {sampling_rate:.1f} Hz
    💗 Detected Peaks: {len(ppg_peaks)} ({peak_quality:.1f} peaks/min)
    📈 PPG SNR: {snr_ppg:.2f} dB
    
    🫀 PHYSIOLOGICAL PARAMETERS
    
    💓 Heart Rate: {np.mean(hr_values):.1f} ± {np.std(hr_values):.1f} BPM
    🔄 HRV (RMSSD): {np.sqrt(np.mean(np.diff(hr_values)**2)) if len(hr_values) > 1 else 0:.2f} ms
    📊 PPG Range: {np.min(ppg_signal):.3f} to {np.max(ppg_signal):.3f}
    """
    
    ax5.text(0.02, 0.95, metrics_text, transform=ax5.transAxes, fontsize=11, 
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(f'{title} - Generated on {timestamp}', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Professional PPG analysis saved to: {save_path}")
    
    plt.show()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Professional PPG Data Visualization')
    parser.add_argument('--data', default='auto', help='Data source: auto, realistic, bidmc, or file path')
    parser.add_argument('--mode', choices=['plot', 'analyze'], default='plot', help='Operation mode')
    parser.add_argument('--save', help='Save plot to file (e.g., professional_ppg.png)')
    
    args = parser.parse_args()
    
    df = None
    
    # Load data based on selection
    if args.data == 'auto' or args.data == 'realistic':
        df = load_realistic_ppg_data()
        title = "Professional PPG Analysis - Realistic Data"
    elif args.data == 'bidmc':
        df = load_bidmc_data()
        title = "Professional PPG Analysis - BIDMC Dataset"
    else:
        # Try to load custom file
        try:
            if args.data.endswith('.csv'):
                df = pd.read_csv(args.data)
                title = f"Professional PPG Analysis - {args.data}"
            else:
                # Assume it's Arduino serial data
                with open(args.data, 'r') as f:
                    data_string = f.read()
                df = parse_arduino_serial_data(data_string)
                title = f"Professional PPG Analysis - Arduino Data"
        except Exception as e:
            print(f"Error loading data file {args.data}: {e}")
            return
    
    if df is None or df.empty:
        print("❌ No valid data found! Please check your data files.")
        print("\nTip: Make sure you have one of these files:")
        print("  - realistic_ecg_ppg_data.csv")
        print("  - bidmc01_ecg.csv and bidmc01_ppg.csv")
        print("  - Or specify a custom data file with --data PATH")
        return
    
    print(f"✅ Successfully loaded {len(df)} samples")
    print(f"📊 Data columns: {list(df.columns)}")
    
    # Create professional visualization
    if args.mode == 'plot':
        save_name = args.save if args.save else f"professional_ppg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        create_professional_ppg_plot(df, title, save_name)
    
    print("🎉 PPG analysis complete!")

if __name__ == "__main__":
    main()
