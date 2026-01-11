#!/usr/bin/env python3
"""
Simple and Accurate Arduino PPG Plotter
=======================================
This script creates clean, accurate PPG visualizations from Arduino data.
Focuses on proper data interpretation and clear visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

def create_simple_ppg_plot(df, title="Arduino PPG Data Analysis", save_path=None):
    """Create simple, accurate PPG visualization"""
    
    print(f"📊 Analyzing Arduino PPG Data:")
    print(f"   - Total samples: {len(df)}")
    print(f"   - Columns: {list(df.columns)}")
    
    # Clean up timestamp issues - skip the first anomalous point
    if len(df) > 1 and abs(df['timestamp'].iloc[1] - df['timestamp'].iloc[0]) > 1000:
        df = df.iloc[1:].copy()  # Skip first sample with timestamp reset
        print(f"   - Skipped first sample, using {len(df)} samples")
    
    # Create time axis in seconds, relative to start
    timestamps_ms = df['timestamp'].values
    time_seconds = (timestamps_ms - timestamps_ms[0]) / 1000.0
    
    # Calculate actual sampling rate
    if len(time_seconds) > 1:
        duration = time_seconds[-1] - time_seconds[0]
        actual_rate = len(df) / duration if duration > 0 else 125
        mean_interval = np.mean(np.diff(timestamps_ms))
        print(f"   - Duration: {duration:.2f} seconds")
        print(f"   - Sampling rate: ~{actual_rate:.1f} Hz")
        print(f"   - Mean interval: {mean_interval:.3f} ms")
    
    # Extract signals
    ecg_data = df['ecg'].values
    ppg_data = df['ppg'].values
    hr_data = df['heart_rate'].values
    
    # Set up the plot
    plt.style.use('default')
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. PPG Signal Plot
    ax1 = axes[0]
    ax1.plot(time_seconds, ppg_data, 'r-', linewidth=1.5, label='PPG Signal')
    
    # Find and mark PPG peaks (simple peak detection)
    try:
        if len(ppg_data) > 10:
            # Simple peak detection for PPG
            min_distance = max(1, int(len(ppg_data) / 20))  # Adaptive distance
            peaks, _ = find_peaks(ppg_data, 
                                distance=min_distance,
                                height=np.mean(ppg_data) + 0.1 * np.std(ppg_data))
            
            if len(peaks) > 0:
                ax1.plot(time_seconds[peaks], ppg_data[peaks], 'bo', 
                        markersize=5, alpha=0.8, label=f'Peaks ({len(peaks)})')
                print(f"   - Detected {len(peaks)} PPG peaks")
            else:
                print(f"   - No clear PPG peaks detected")
    except:
        print(f"   - Peak detection failed")
    
    ax1.set_title('PPG (Photoplethysmogram) Signal', fontsize=14, fontweight='bold')
    ax1.set_ylabel('PPG Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, max(1.6, np.max(ppg_data) * 1.1)])
    
    # Add PPG statistics box
    ppg_mean = np.mean(ppg_data)
    ppg_std = np.std(ppg_data)
    ppg_min = np.min(ppg_data)
    ppg_max = np.max(ppg_data)
    
    stats_text = f'PPG Stats: Mean={ppg_mean:.3f}, Std={ppg_std:.3f}, Range=[{ppg_min:.2f}, {ppg_max:.2f}]'
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    # 2. ECG Signal Plot
    ax2 = axes[1]
    ax2.plot(time_seconds, ecg_data, 'g-', linewidth=1.5, label='ECG Signal')
    
    # ECG peak detection
    try:
        if len(ecg_data) > 10:
            min_distance = max(1, int(len(ecg_data) / 20))
            ecg_peaks, _ = find_peaks(ecg_data, 
                                    distance=min_distance,
                                    height=np.mean(ecg_data) + 0.5 * np.std(ecg_data))
            
            if len(ecg_peaks) > 0:
                ax2.plot(time_seconds[ecg_peaks], ecg_data[ecg_peaks], 'ro', 
                        markersize=5, alpha=0.8, label=f'R-peaks ({len(ecg_peaks)})')
                print(f"   - Detected {len(ecg_peaks)} ECG peaks")
    except:
        print(f"   - ECG peak detection failed")
    
    ax2.set_title('ECG (Electrocardiogram) Reference Signal', fontsize=14, fontweight='bold')
    ax2.set_ylabel('ECG Amplitude')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add ECG statistics
    ecg_mean = np.mean(ecg_data)
    ecg_std = np.std(ecg_data)
    ecg_min = np.min(ecg_data)
    ecg_max = np.max(ecg_data)
    
    ecg_stats_text = f'ECG Stats: Mean={ecg_mean:.1f}, Std={ecg_std:.1f}, Range=[{ecg_min:.0f}, {ecg_max:.0f}]'
    ax2.text(0.02, 0.95, ecg_stats_text, transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # 3. Heart Rate Plot
    ax3 = axes[2]
    ax3.plot(time_seconds, hr_data, 'b-', linewidth=2, marker='o', markersize=3,
             alpha=0.8, label='Arduino Heart Rate')
    
    # Add heart rate statistics
    hr_mean = np.mean(hr_data)
    hr_std = np.std(hr_data)
    
    ax3.axhline(y=hr_mean, color='red', linestyle='--', alpha=0.7, 
               label=f'Mean: {hr_mean:.1f} BPM')
    
    if hr_std > 0.1:  # Only show variability if there is some
        ax3.fill_between(time_seconds, hr_mean - hr_std, hr_mean + hr_std, 
                        alpha=0.2, color='blue', label=f'±1σ ({hr_std:.1f})')
    
    ax3.set_title('Heart Rate Estimation', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Heart Rate (BPM)')
    ax3.set_xlabel('Time (seconds)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([55, 65])  # Reasonable range around 60 BPM
    
    # Set common x-axis limits
    for ax in axes:
        ax.set_xlim([time_seconds[0], time_seconds[-1]])
    
    # Add comprehensive information box
    info_text = f"""Arduino PPG Analysis Results:
📊 Dataset: {len(df)} samples over {duration:.2f} seconds (~{actual_rate:.1f} Hz)
💗 PPG Range: {ppg_min:.3f} to {ppg_max:.3f} (TinyML prediction)
💓 Heart Rate: {hr_mean:.1f} BPM (constant from Arduino)
🔧 Source: Arduino Nano 33 BLE with CSV playback
📈 ECG→PPG: LSTM TinyML model inference"""
    
    fig.text(0.02, 0.02, info_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for info box
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Arduino PPG plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n📋 Analysis Summary:")
    print(f"   - PPG Signal: {ppg_min:.3f} to {ppg_max:.3f} (mean: {ppg_mean:.3f})")
    print(f"   - ECG Signal: {ecg_min:.0f} to {ecg_max:.0f} (mean: {ecg_mean:.1f})")
    print(f"   - Heart Rate: {hr_mean:.1f} BPM (constant)")
    print(f"   - Recording: {duration:.2f}s at ~{actual_rate:.1f} Hz")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Simple Arduino PPG Analysis')
    parser.add_argument('--data', required=True, help='Arduino CSV data file')
    parser.add_argument('--save', help='Save plot to file')
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = pd.read_csv(args.data)
        print(f"✅ Loaded: {args.data}")
        
        # Generate plot
        save_name = args.save or f"arduino_ppg_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        title = f"Arduino PPG Analysis - {args.data.split('/')[-1]}"
        
        create_simple_ppg_plot(df, title, save_name)
        
        print(f"🎉 Simple Arduino PPG analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
