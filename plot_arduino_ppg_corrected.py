#!/usr/bin/env python3
"""
Corrected Arduino PPG Data Plotter
==================================
This script creates accurate PPG visualizations from Arduino serial data.
Handles real Arduino timing and data formats correctly.

Usage:
python3 plot_arduino_ppg_corrected.py [--data PATH] [--save FILENAME]
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys
from datetime import datetime
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

def analyze_arduino_data(df):
    """Analyze Arduino data structure and fix timing issues"""
    print(f"📊 Data Analysis:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Columns: {list(df.columns)}")
    
    if 'timestamp' in df.columns:
        timestamps = df['timestamp'].values
        print(f"  - Time range: {timestamps.min():.3f} to {timestamps.max():.3f} ms")
        
        # Calculate actual sampling intervals
        if len(timestamps) > 1:
            # Remove first timestamp jump if it exists
            if timestamps[1] - timestamps[0] > 1000:  # Large jump indicates reset
                print(f"  - Detected timestamp reset after first sample")
                timestamps = timestamps[1:].copy()  # Skip first sample
                df = df.iloc[1:].copy()
                df['timestamp'] = timestamps
                
            # Calculate intervals between consecutive samples
            intervals = np.diff(timestamps)
            mean_interval = np.mean(intervals)
            actual_rate = 1000.0 / mean_interval  # Convert ms to Hz
            
            print(f"  - Mean interval: {mean_interval:.3f} ms")
            print(f"  - Actual sampling rate: {actual_rate:.1f} Hz")
            print(f"  - Duration: {(timestamps[-1] - timestamps[0])/1000:.2f} seconds")
            
            return df, actual_rate
    
    # Fallback if no timestamp analysis possible
    print("  - Using default 125 Hz sampling rate")
    return df, 125.0

def filter_ppg_signal(ppg_data, fs=125, lowcut=0.5, highcut=8.0):
    """Apply proper PPG filtering"""
    if len(ppg_data) < 4:
        return ppg_data.copy()
    
    try:
        nyquist = 0.5 * fs
        low = max(lowcut / nyquist, 0.001)
        high = min(highcut / nyquist, 0.999)
        
        b, a = butter(2, [low, high], btype='band')  # Lower order for stability
        filtered_signal = filtfilt(b, a, ppg_data)
        return filtered_signal
    except:
        print("  - Warning: Filtering failed, using raw signal")
        return ppg_data.copy()

def detect_ppg_peaks_corrected(ppg_data, fs=125):
    """Detect PPG peaks with corrected parameters"""
    if len(ppg_data) < 10:
        return np.array([]), ppg_data
    
    # Filter the signal
    filtered_ppg = filter_ppg_signal(ppg_data, fs)
    
    # PPG peaks should be the systolic peaks (maximum values)
    # Set minimum distance based on reasonable heart rate (40-180 BPM)
    min_distance = int(fs * 60 / 180)  # Minimum distance for 180 BPM
    max_distance = int(fs * 60 / 40)   # Maximum distance for 40 BPM
    
    # Find peaks with adaptive threshold
    prominence_threshold = np.std(filtered_ppg) * 0.2
    height_threshold = np.mean(filtered_ppg) + np.std(filtered_ppg) * 0.1
    
    peaks, properties = find_peaks(
        filtered_ppg,
        distance=min_distance,
        prominence=prominence_threshold,
        height=height_threshold
    )
    
    print(f"  - Detected {len(peaks)} PPG peaks")
    return peaks, filtered_ppg

def calculate_heart_rate_corrected(peaks, fs=125):
    """Calculate heart rate from peaks correctly"""
    if len(peaks) < 2:
        return np.array([60.0])
    
    # Calculate intervals between peaks (in seconds)
    intervals = np.diff(peaks) / fs
    
    # Convert to heart rates (beats per minute)
    heart_rates = 60.0 / intervals
    
    # Remove outliers (physiologically impossible heart rates)
    heart_rates = heart_rates[(heart_rates >= 40) & (heart_rates <= 200)]
    
    if len(heart_rates) == 0:
        return np.array([60.0])
    
    return heart_rates

def create_corrected_ppg_plot(df, title="Arduino PPG Analysis", save_path=None):
    """Create corrected PPG visualization"""
    
    # Analyze the data first
    df_clean, actual_fs = analyze_arduino_data(df)
    
    # Set up the plot
    plt.style.use('default')
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Extract data
    if 'timestamp' in df_clean.columns:
        timestamps = df_clean['timestamp'].values
        time_seconds = (timestamps - timestamps[0]) / 1000.0  # Convert to seconds from start
    else:
        time_seconds = np.arange(len(df_clean)) / actual_fs
    
    ecg_data = df_clean['ecg'].values
    ppg_data = df_clean['ppg'].values
    hr_data = df_clean['heart_rate'].values
    
    # 1. PPG Signal Analysis
    ax1 = axes[0]
    ax1.plot(time_seconds, ppg_data, 'r-', linewidth=1.5, alpha=0.8, label='Raw PPG')
    
    # Filter and detect peaks
    peaks, ppg_filtered = detect_ppg_peaks_corrected(ppg_data, actual_fs)
    ax1.plot(time_seconds, ppg_filtered, 'darkred', linewidth=2, label='Filtered PPG')
    
    # Mark detected peaks
    if len(peaks) > 0:
        ax1.plot(time_seconds[peaks], ppg_data[peaks], 'bo', markersize=6, 
                alpha=0.8, label=f'Detected Peaks ({len(peaks)})')
    
    ax1.set_title('PPG Signal with Peak Detection', fontsize=14, fontweight='bold')
    ax1.set_ylabel('PPG Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, max(1.6, np.max(ppg_data) * 1.1)])
    
    # Add PPG statistics
    ppg_stats = (f'Mean: {np.mean(ppg_data):.3f}, Std: {np.std(ppg_data):.3f}, '
                f'Range: {np.min(ppg_data):.2f} - {np.max(ppg_data):.2f}')
    ax1.text(0.02, 0.95, ppg_stats, transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # 2. ECG Signal
    ax2 = axes[1]
    ax2.plot(time_seconds, ecg_data, 'g-', linewidth=1.5, alpha=0.8, label='ECG Signal')
    
    # Filter ECG
    try:
        ecg_filtered = filter_ppg_signal(ecg_data, actual_fs, lowcut=1.0, highcut=30.0)
        ax2.plot(time_seconds, ecg_filtered, 'darkgreen', linewidth=2, alpha=0.8, label='Filtered ECG')
    except:
        ecg_filtered = ecg_data
    
    ax2.set_title('ECG Reference Signal', fontsize=14, fontweight='bold')
    ax2.set_ylabel('ECG Amplitude')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Heart Rate Analysis
    ax3 = axes[2]
    
    # Calculate heart rate from detected peaks
    calculated_hr = calculate_heart_rate_corrected(peaks, actual_fs)
    
    # Plot Arduino's heart rate
    ax3.plot(time_seconds, hr_data, 'b-', linewidth=2, marker='o', markersize=4, 
             alpha=0.8, label='Arduino HR Estimate')
    
    # Plot calculated heart rate from peaks
    if len(calculated_hr) > 0:
        peak_times = time_seconds[peaks[1:len(calculated_hr)+1]]
        ax3.plot(peak_times, calculated_hr, 'r-', linewidth=2, marker='s', markersize=4,
                alpha=0.8, label='Calculated from Peaks')
        
        # Add mean line
        mean_hr = np.mean(calculated_hr)
        ax3.axhline(y=mean_hr, color='red', linestyle='--', alpha=0.7,
                   label=f'Mean: {mean_hr:.1f} BPM')
    
    ax3.set_title('Heart Rate Analysis', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Heart Rate (BPM)')
    ax3.set_ylim([50, 80])
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Signal Quality and Statistics
    ax4 = axes[3]
    ax4.axis('off')
    
    # Calculate comprehensive statistics
    duration = time_seconds[-1] - time_seconds[0]
    num_samples = len(df_clean)
    
    # Peak analysis
    if len(peaks) > 1:
        avg_peak_interval = np.mean(np.diff(peaks)) / actual_fs
        estimated_hr_from_peaks = 60 / avg_peak_interval
        peak_quality = f"{len(peaks)} peaks, Est. HR: {estimated_hr_from_peaks:.1f} BPM"
    else:
        peak_quality = "Insufficient peaks detected"
    
    # Signal quality metrics
    ppg_snr = np.mean(ppg_data) / np.std(ppg_data) if np.std(ppg_data) > 0 else 0
    
    stats_text = f"""
📊 ARDUINO PPG ANALYSIS SUMMARY
    
⏱️  Recording Duration: {duration:.2f} seconds
📡 Sampling Rate: {actual_fs:.1f} Hz ({num_samples:,} samples)
💗 Peak Detection: {peak_quality}
📈 PPG SNR: {ppg_snr:.2f}

🫀 PHYSIOLOGICAL MEASUREMENTS

💓 Arduino HR Reading: {np.mean(hr_data):.1f} BPM (constant)
🔍 PPG Peak Analysis: {len(calculated_hr)} intervals analyzed
📊 PPG Amplitude Range: {np.min(ppg_data):.3f} to {np.max(ppg_data):.3f}
📈 ECG Amplitude Range: {np.min(ecg_data):.0f} to {np.max(ecg_data):.0f}

🔧 TECHNICAL DETAILS

⚙️  Data Source: Arduino Nano 33 BLE (CSV Playback Mode)
🎯 TinyML Model: LSTM ECG→PPG Prediction  
📊 Peak Detection: Adaptive threshold with {actual_fs:.0f}Hz filtering
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
    
    # Set common x-axis
    for ax in axes[:3]:
        ax.set_xlim([time_seconds[0], time_seconds[-1]])
        ax.set_xlabel('Time (seconds)')
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.99, 0.01, f'Generated: {timestamp}', ha='right', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Corrected PPG analysis saved to: {save_path}")
    
    plt.show()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Corrected Arduino PPG Analysis')
    parser.add_argument('--data', required=True, help='Arduino CSV data file path')
    parser.add_argument('--save', help='Save plot to file (e.g., corrected_ppg.png)')
    
    args = parser.parse_args()
    
    try:
        # Load the data
        df = pd.read_csv(args.data)
        print(f"✅ Loaded Arduino data from: {args.data}")
        
        # Create corrected visualization
        save_name = args.save if args.save else f"corrected_arduino_ppg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        title = f"Corrected Arduino PPG Analysis - {args.data.split('/')[-1]}"
        
        create_corrected_ppg_plot(df, title, save_name)
        
        print("🎉 Corrected PPG analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
