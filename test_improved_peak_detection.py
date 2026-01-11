#!/usr/bin/env python3
"""
Test and Visualize Improved ECG Peak Detection

This script specifically tests the improved peak detection algorithm
and shows how it better captures ECG R-peaks for PPG pulse generation.

Usage:
    python test_improved_peak_detection.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from train_improved_ppg_model import load_ecg_data, generate_ppg_from_ecg

def old_peak_detection(ecg_data):
    """Original peak detection method for comparison"""
    peaks, _ = find_peaks(ecg_data, 
                         height=np.mean(ecg_data) + 0.4*np.std(ecg_data), 
                         distance=250)
    return peaks

def new_peak_detection(ecg_data):
    """Improved peak detection method"""
    # Apply light smoothing to reduce noise
    smoothed_ecg = gaussian_filter1d(ecg_data, sigma=1.0)
    
    # Calculate dynamic threshold based on signal characteristics
    signal_mean = np.mean(smoothed_ecg)
    signal_std = np.std(smoothed_ecg)
    
    # Use a more sensitive threshold to catch more peaks
    base_threshold = signal_mean + 0.2 * signal_std  # Reduced from 0.4 to 0.2
    
    # Estimate heart rate to set appropriate distance parameter
    min_distance = int(360 * 60 / 150)  # Max 150 BPM = 144 samples minimum distance
    max_distance = int(360 * 60 / 50)   # Min 50 BPM = 432 samples maximum expected
    
    # First pass: detect peaks with lower threshold
    peaks_candidates, properties = find_peaks(
        smoothed_ecg,
        height=base_threshold,
        distance=min_distance,
        prominence=signal_std * 0.1,
        width=3
    )
    
    # Second pass: refine peaks by checking for missing peaks in large gaps
    refined_peaks = []
    
    if len(peaks_candidates) > 0:
        refined_peaks.append(peaks_candidates[0])
        
        for i in range(1, len(peaks_candidates)):
            prev_peak = refined_peaks[-1]
            current_peak = peaks_candidates[i]
            gap = current_peak - prev_peak
            
            # If gap is too large, look for missing peaks
            if gap > max_distance * 0.8:
                # Search for additional peaks in the gap
                gap_start = prev_peak + min_distance
                gap_end = current_peak - min_distance
                
                if gap_end > gap_start:
                    gap_signal = smoothed_ecg[gap_start:gap_end]
                    gap_threshold = signal_mean + 0.1 * signal_std
                    
                    additional_peaks, _ = find_peaks(
                        gap_signal,
                        height=gap_threshold,
                        distance=min_distance // 2,
                        prominence=signal_std * 0.05
                    )
                    
                    if len(additional_peaks) > 0:
                        additional_peaks = additional_peaks + gap_start
                        peak_heights = smoothed_ecg[additional_peaks]
                        if len(additional_peaks) > 2:
                            keep_indices = np.argsort(peak_heights)[-2:]
                            additional_peaks = additional_peaks[keep_indices]
                        
                        refined_peaks.extend(sorted(additional_peaks))
            
            refined_peaks.append(current_peak)
    
    peaks = np.array(refined_peaks)
    
    # Final validation: remove peaks that are too close together
    if len(peaks) > 1:
        valid_peaks = [peaks[0]]
        for i in range(1, len(peaks)):
            if peaks[i] - valid_peaks[-1] >= min_distance:
                valid_peaks.append(peaks[i])
        peaks = np.array(valid_peaks)
    
    return peaks

def compare_peak_detection():
    """Compare old vs new peak detection methods"""
    print("Loading ECG data for peak detection comparison...")
    ecg = load_ecg_data()
    
    # Use a subset for visualization (first 10,000 samples = ~28 seconds at 360 Hz)
    subset_size = 10000
    ecg_subset = ecg[:subset_size]
    
    print(f"Analyzing {subset_size} ECG samples ({subset_size/360:.1f} seconds)")
    
    # Apply both methods
    old_peaks = old_peak_detection(ecg_subset)
    new_peaks = new_peak_detection(ecg_subset)
    
    print(f"Old method detected: {len(old_peaks)} peaks")
    print(f"New method detected: {len(new_peaks)} peaks")
    print(f"Improvement: {len(new_peaks) - len(old_peaks)} additional peaks detected")
    
    # Calculate expected heart rate range
    duration_min = subset_size / 360 / 60  # Duration in minutes
    expected_min_beats = int(50 * duration_min)  # 50 BPM minimum
    expected_max_beats = int(150 * duration_min)  # 150 BPM maximum
    
    print(f"\nExpected beats for {duration_min:.1f} minutes: {expected_min_beats}-{expected_max_beats}")
    print(f"Old method capture rate: {len(old_peaks)/expected_max_beats*100:.1f}% of maximum expected")
    print(f"New method capture rate: {len(new_peaks)/expected_max_beats*100:.1f}% of maximum expected")
    
    return ecg_subset, old_peaks, new_peaks

def visualize_peak_comparison():
    """Create comprehensive visualization comparing peak detection methods"""
    ecg_subset, old_peaks, new_peaks = compare_peak_detection()
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # Time axis for plotting
    time_samples = np.arange(len(ecg_subset))
    
    # 1. Original ECG with old peak detection
    axes[0].plot(time_samples, ecg_subset, 'b-', linewidth=1, alpha=0.7, label='ECG Signal')
    axes[0].scatter(old_peaks, ecg_subset[old_peaks], color='red', s=30, zorder=5, 
                   label=f'Old Detection ({len(old_peaks)} peaks)')
    axes[0].set_title('Original Peak Detection Method', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('ECG Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. ECG with new peak detection
    axes[1].plot(time_samples, ecg_subset, 'b-', linewidth=1, alpha=0.7, label='ECG Signal')
    axes[1].scatter(new_peaks, ecg_subset[new_peaks], color='green', s=30, zorder=5,
                   label=f'New Detection ({len(new_peaks)} peaks)')
    axes[1].set_title('Improved Peak Detection Method', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('ECG Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Comparison overlay
    axes[2].plot(time_samples, ecg_subset, 'b-', linewidth=1, alpha=0.5, label='ECG Signal')
    axes[2].scatter(old_peaks, ecg_subset[old_peaks], color='red', s=40, zorder=5,
                   alpha=0.7, label=f'Old ({len(old_peaks)} peaks)')
    axes[2].scatter(new_peaks, ecg_subset[new_peaks], color='green', s=25, zorder=6,
                   marker='x', linewidth=2, label=f'New ({len(new_peaks)} peaks)')
    axes[2].set_title('Peak Detection Comparison (Red: Old, Green X: New)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('ECG Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. Generate PPG with improved detection and show correspondence
    print("Generating PPG with improved peak detection...")
    ppg_signal = generate_ppg_from_ecg(ecg_subset)
    
    # Normalize for visualization
    ecg_norm = (ecg_subset - ecg_subset.mean()) / ecg_subset.std()
    ppg_norm = (ppg_signal - ppg_signal.mean()) / ppg_signal.std()
    
    axes[3].plot(time_samples, ecg_norm, 'b-', linewidth=1, alpha=0.7, label='ECG (normalized)')
    axes[3].plot(time_samples, ppg_norm, 'r-', linewidth=1.5, alpha=0.8, label='Generated PPG')
    axes[3].scatter(new_peaks, ecg_norm[new_peaks], color='green', s=25, zorder=5,
                   marker='o', label='Detected R-peaks')
    
    # Show expected PPG peaks (with delay)
    ppg_peak_positions = new_peaks + 120  # 120 sample delay
    valid_ppg_peaks = ppg_peak_positions[ppg_peak_positions < len(ppg_norm)]
    if len(valid_ppg_peaks) > 0:
        axes[3].scatter(valid_ppg_peaks, ppg_norm[valid_ppg_peaks], color='orange', s=20, 
                       marker='s', zorder=5, alpha=0.8, label='Corresponding PPG pulses')
    
    axes[3].set_title('ECG R-peaks and Corresponding PPG Pulses', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Sample Index')
    axes[3].set_ylabel('Normalized Amplitude')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comparison
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"peak_detection_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPeak detection comparison saved to: {filename}")
    
    plt.show()
    
    return len(old_peaks), len(new_peaks)

def detailed_section_analysis():
    """Analyze a small section in detail to show missed peaks"""
    print("\n" + "="*60)
    print("DETAILED SECTION ANALYSIS")
    print("="*60)
    
    ecg = load_ecg_data()
    
    # Analyze a 5-second section (1800 samples)
    start_idx = 5000
    section_size = 1800
    ecg_section = ecg[start_idx:start_idx + section_size]
    
    old_peaks = old_peak_detection(ecg_section)
    new_peaks = new_peak_detection(ecg_section)
    
    print(f"5-second section analysis (samples {start_idx}-{start_idx + section_size}):")
    print(f"Old method: {len(old_peaks)} peaks")
    print(f"New method: {len(new_peaks)} peaks")
    
    # Expected beats in 5 seconds
    expected_beats_5sec = "4-12 beats (50-150 BPM)"
    print(f"Expected range: {expected_beats_5sec}")
    
    # Create detailed plot
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    time_axis = np.arange(len(ecg_section))
    
    # ECG with old detection
    axes[0].plot(time_axis, ecg_section, 'b-', linewidth=2, alpha=0.8)
    if len(old_peaks) > 0:
        axes[0].scatter(old_peaks, ecg_section[old_peaks], color='red', s=50, zorder=5)
        axes[0].set_title(f'Old Detection: {len(old_peaks)} peaks in 5 seconds', fontsize=14)
    else:
        axes[0].set_title('Old Detection: No peaks found in 5 seconds', fontsize=14)
    axes[0].set_ylabel('ECG Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # ECG with new detection
    axes[1].plot(time_axis, ecg_section, 'b-', linewidth=2, alpha=0.8)
    if len(new_peaks) > 0:
        axes[1].scatter(new_peaks, ecg_section[new_peaks], color='green', s=50, zorder=5)
        axes[1].set_title(f'New Detection: {len(new_peaks)} peaks in 5 seconds', fontsize=14)
    else:
        axes[1].set_title('New Detection: No peaks found in 5 seconds', fontsize=14)
    axes[1].set_ylabel('ECG Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # PPG generated with new method
    ppg_section = generate_ppg_from_ecg(ecg_section)
    ecg_norm = (ecg_section - ecg_section.mean()) / ecg_section.std()
    ppg_norm = (ppg_section - ppg_section.mean()) / ppg_section.std()
    
    axes[2].plot(time_axis, ecg_norm, 'b-', linewidth=1.5, alpha=0.7, label='ECG')
    axes[2].plot(time_axis, ppg_norm, 'r-', linewidth=2, alpha=0.8, label='Generated PPG')
    if len(new_peaks) > 0:
        axes[2].scatter(new_peaks, ecg_norm[new_peaks], color='green', s=40, 
                       zorder=5, marker='o', label='R-peaks')
    axes[2].set_title('ECG and Generated PPG with Peak Correspondence', fontsize=14)
    axes[2].set_xlabel('Sample Index (5 seconds = 1800 samples at 360 Hz)')
    axes[2].set_ylabel('Normalized Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save detailed analysis
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detailed_peak_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Detailed analysis saved to: {filename}")
    
    plt.show()

def main():
    """Run the complete peak detection analysis"""
    print("🔍 ECG Peak Detection Improvement Analysis")
    print("=" * 50)
    
    try:
        # Compare detection methods
        old_count, new_count = visualize_peak_comparison()
        
        # Show improvement summary
        print(f"\n📊 IMPROVEMENT SUMMARY:")
        print(f"   Old method: {old_count} peaks detected")
        print(f"   New method: {new_count} peaks detected")
        print(f"   Improvement: +{new_count - old_count} peaks ({(new_count-old_count)/old_count*100:.1f}% increase)")
        
        # Detailed section analysis
        detailed_section_analysis()
        
        print(f"\n✅ Analysis complete! The improved peak detection should now capture")
        print(f"   more ECG R-peaks, resulting in better PPG pulse correspondence.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please ensure your ECG data file is available.")

if __name__ == "__main__":
    main()
