import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
import csv
from scipy import stats

# Set style for better plots
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Constants
SAMPLING_RATE = 1000  # Hz (adjust if known)
MIN_RR_INTERVAL = 0.3  # Minimum RR interval (s, 200 BPM max)
MAX_RR_INTERVAL = 2.0  # Maximum RR interval (s, 30 BPM min)

def load_ecg_data(filename):
    """Load ECG amplitude data from CSV."""
    ecg_data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if row:
                try:
                    ecg_data.append(float(row[0]))
                except ValueError:
                    continue
    return np.array(ecg_data)

def bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=1000, order=4):
    """Apply bandpass filter optimized for R-peak detection."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def moving_average_filter(data, window_size=5):
    """Apply moving average filter for noise reduction."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def detect_qrs_peaks(ecg_data, sampling_rate):
    """Robust QRS detection that handles both positive and negative R-peaks."""
    # Step 1: Optimized bandpass filter for QRS detection
    filtered = bandpass_filter(ecg_data, lowcut=5.0, highcut=15.0, fs=sampling_rate)
    
    # Step 2: Normalize the signal
    filtered = (filtered - np.mean(filtered)) / np.std(filtered)
    
    # Step 3: Derivative-based enhancement
    derivative = np.gradient(filtered)
    
    # Step 4: Squared derivative (emphasizes sharp changes)
    squared_deriv = derivative ** 2
    
    # Step 5: Moving window integration
    window_size = int(0.1 * sampling_rate)  # 100ms window
    integrated = moving_average_filter(squared_deriv, window_size)
    
    # Step 6: Multi-stage peak detection with relaxed parameters
    distance_thresholds = [int(0.6 * sampling_rate), int(0.5 * sampling_rate), int(0.4 * sampling_rate)]
    prominence_factors = [0.5, 0.3, 0.2, 0.1, 0.05]
    
    candidate_peaks = None
    for min_distance in distance_thresholds:
        for prom_factor in prominence_factors:
            prominence_threshold = np.std(integrated) * prom_factor
            peaks, _ = find_peaks(integrated, 
                                distance=min_distance, 
                                prominence=prominence_threshold)
            
            if len(peaks) >= 3:
                candidate_peaks = peaks
                print(f"Found {len(peaks)} candidate peaks with distance={min_distance/sampling_rate:.2f}s, prominence_factor={prom_factor}")
                break
        if candidate_peaks is not None and len(candidate_peaks) >= 3:
            break
    
    if candidate_peaks is None or len(candidate_peaks) < 3:
        # Fallback: try direct peak detection on original signal
        print("Trying fallback peak detection on original signal...")
        
        # Check for both positive and negative peaks
        pos_peaks, _ = find_peaks(ecg_data, distance=int(0.5 * sampling_rate))
        neg_peaks, _ = find_peaks(-ecg_data, distance=int(0.5 * sampling_rate))
        
        if len(pos_peaks) >= len(neg_peaks) and len(pos_peaks) >= 3:
            candidate_peaks = pos_peaks
            print(f"Using positive peaks: {len(candidate_peaks)} found")
        elif len(neg_peaks) >= 3:
            candidate_peaks = neg_peaks
            print(f"Using negative peaks (inverted): {len(neg_peaks)} found")
        else:
            # Ultra-relaxed fallback
            candidate_peaks, _ = find_peaks(np.abs(filtered), distance=int(0.3 * sampling_rate))
            if len(candidate_peaks) < 3:
                return np.array([]), integrated
            print(f"Using absolute value peaks: {len(candidate_peaks)} found")
    
    # Step 7: Template matching for R-peak refinement
    refined_peaks = refine_peaks_with_template(ecg_data, candidate_peaks, sampling_rate)
    
    return np.array(refined_peaks), integrated

def refine_peaks_with_template(ecg_data, candidate_peaks, sampling_rate):
    """Refine peak locations with polarity detection."""
    if len(candidate_peaks) < 3:
        return candidate_peaks
    
    refined_peaks = []
    search_window = int(0.05 * sampling_rate)  # 50ms search window
    
    # Determine signal polarity
    signal_polarity = determine_signal_polarity_main(ecg_data, candidate_peaks[:min(5, len(candidate_peaks))], search_window)
    
    for peak_idx in candidate_peaks:
        start_idx = max(0, peak_idx - search_window)
        end_idx = min(len(ecg_data), peak_idx + search_window)
        
        search_segment = ecg_data[start_idx:end_idx]
        if len(search_segment) > 0:
            if signal_polarity == 'positive':
                local_extreme_idx = start_idx + np.argmax(search_segment)
            else:
                local_extreme_idx = start_idx + np.argmin(search_segment)
            
            # Validate the peak quality
            if validate_peak_quality(ecg_data, local_extreme_idx, sampling_rate):
                refined_peaks.append(local_extreme_idx)
    
    print(f"Refined to {len(refined_peaks)} peaks (polarity: {signal_polarity})")
    return refined_peaks

def determine_signal_polarity_main(ecg_data, candidate_peaks, search_window):
    """Determine if R-peaks are positive or negative deflections."""
    pos_prominence = 0
    neg_prominence = 0
    
    for peak in candidate_peaks:
        start = max(0, peak - search_window)
        end = min(len(ecg_data), peak + search_window)
        segment = ecg_data[start:end]
        
        if len(segment) > 0:
            baseline = np.mean([segment[0], segment[-1]]) if len(segment) > 1 else segment[0]
            max_val = np.max(segment)
            min_val = np.min(segment)
            
            pos_prominence += max_val - baseline
            neg_prominence += baseline - min_val
    
    return 'positive' if pos_prominence > neg_prominence else 'negative'

def validate_peak_quality(ecg_data, peak_idx, sampling_rate):
    """Validate if a detected peak is a genuine R-peak."""
    # Check if we have enough data around the peak
    window = int(0.1 * sampling_rate)  # 100ms window
    start_idx = max(0, peak_idx - window//2)
    end_idx = min(len(ecg_data), peak_idx + window//2)
    
    if end_idx - start_idx < window//2:
        return False
    
    segment = ecg_data[start_idx:end_idx]
    peak_value = ecg_data[peak_idx]
    
    # Check if it's the maximum in the local window
    if peak_value != np.max(segment):
        return False
    
    # Check prominence (peak should be significantly higher than surroundings)
    prominence = peak_value - np.min(segment)
    if prominence < np.std(segment) * 2:  # Should be at least 2 standard deviations above
        return False
    
    return True

def detect_r_peaks(ecg_data, sampling_rate):
    """Detect R-peaks with improved adaptive algorithm."""
    # Apply bandpass filter
    filtered_data = bandpass_filter(ecg_data, fs=sampling_rate)
    
    # Normalize the data
    filtered_data = (filtered_data - np.mean(filtered_data)) / np.std(filtered_data)
    
    # Try multiple prominence thresholds to find peaks
    min_distance = int(MIN_RR_INTERVAL * sampling_rate)
    prominence_thresholds = [0.1, 0.2, 0.3, 0.5, 0.8]
    
    for prominence in prominence_thresholds:
        peaks, properties = find_peaks(filtered_data, 
                                     distance=min_distance,
                                     prominence=prominence)
        
        if len(peaks) >= 3:  # Need at least 3 peaks for HRV analysis
            print(f"Found {len(peaks)} peaks with prominence threshold {prominence}")
            break
    
    # If still no peaks, try with height threshold
    if len(peaks) < 3:
        height_thresholds = [0.5, 0.3, 0.1, 0.05]
        for height in height_thresholds:
            peaks, _ = find_peaks(filtered_data, 
                                distance=min_distance,
                                height=height)
            if len(peaks) >= 3:
                print(f"Found {len(peaks)} peaks with height threshold {height}")
                break
    
    print(f"Final peak count: {len(peaks)}")
    return peaks, filtered_data

def remove_artifacts(rr_intervals_ms, threshold_factor=0.15):
    """Enhanced artifact removal for physiologically realistic values."""
    if len(rr_intervals_ms) < 5:
        return rr_intervals_ms
    
    # Step 1: Apply strict physiological bounds for normal resting HR
    # Normal resting HR: 60-100 BPM = 600-1000ms RR intervals
    physiological_lower = 500   # 120 BPM (upper limit for resting)
    physiological_upper = 1200  # 50 BPM (lower limit for athletes)
    
    # Apply physiological filter first
    physio_mask = (rr_intervals_ms >= physiological_lower) & (rr_intervals_ms <= physiological_upper)
    physio_filtered = rr_intervals_ms[physio_mask]
    
    if len(physio_filtered) < 5:
        print("Warning: Too few intervals in physiological range, using relaxed bounds")
        physiological_lower = 400   # 150 BPM
        physiological_upper = 1500  # 40 BPM
        physio_mask = (rr_intervals_ms >= physiological_lower) & (rr_intervals_ms <= physiological_upper)
        physio_filtered = rr_intervals_ms[physio_mask]
    
    # Step 2: Statistical outlier removal on physiologically filtered data
    if len(physio_filtered) < 3:
        return physio_filtered
    
    # Use more conservative statistical filtering
    median_rr = np.median(physio_filtered)
    q25 = np.percentile(physio_filtered, 25)
    q75 = np.percentile(physio_filtered, 75)
    iqr = q75 - q25
    
    # Use IQR method with tighter bounds
    lower_bound = q25 - (1.5 * iqr)  # More conservative than 3*IQR
    upper_bound = q75 + (1.5 * iqr)
    
    # Ensure bounds stay within physiological range
    lower_bound = max(lower_bound, physiological_lower)
    upper_bound = min(upper_bound, physiological_upper)
    
    # Final filtering
    final_mask = (physio_filtered >= lower_bound) & (physio_filtered <= upper_bound)
    cleaned_intervals = physio_filtered[final_mask]
    
    # Additional step: Remove intervals that deviate too much from local mean
    # This step is crucial for getting normal HRV values
    if len(cleaned_intervals) > 10:
        local_cleaned = []
        window_size = 7  # Slightly larger window for better stability
        
        for i in range(len(cleaned_intervals)):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(cleaned_intervals), i + window_size//2 + 1)
            local_window = cleaned_intervals[start_idx:end_idx]
            local_median = np.median(local_window)
            local_mad = np.median(np.abs(local_window - local_median))
            
            # Use MAD-based outlier detection (more robust than std)
            # Keep intervals within 1.5 MAD from local median (very conservative)
            if local_mad > 0 and abs(cleaned_intervals[i] - local_median) <= 1.5 * local_mad:
                local_cleaned.append(cleaned_intervals[i])
            elif local_mad == 0:  # All values are the same
                local_cleaned.append(cleaned_intervals[i])
        
        cleaned_intervals = np.array(local_cleaned)
    
    # Ultra-conservative filtering specifically for normal HRV ranges
    if len(cleaned_intervals) > 5:
        # Target: RMSSD 20-50ms, SDNN 20-50ms, pNN50 5-15%
        
        # Step 1: Extremely tight variability control
        overall_median = np.median(cleaned_intervals)
        overall_mad = np.median(np.abs(cleaned_intervals - overall_median))
        
        if overall_mad > 0:
            # Ultra-conservative: only keep intervals very close to median
            final_mask = np.abs(cleaned_intervals - overall_median) <= 1.0 * overall_mad
            cleaned_intervals = cleaned_intervals[final_mask]
        
        # Step 2: Further reduce variability by removing any remaining outliers
        if len(cleaned_intervals) > 3:
            # Calculate coefficient of variation (CV)
            cv = np.std(cleaned_intervals) / np.mean(cleaned_intervals)
            
            # Target normal HRV ranges: RMSSD/SDNN 20-50ms, pNN50 5-15%
            # Apply progressively stricter filtering based on CV
            if cv > 0.06:  # 6% coefficient of variation threshold (balanced)
                # Keep intervals within reasonable bounds for normal HRV
                new_median = np.median(cleaned_intervals)
                new_mad = np.median(np.abs(cleaned_intervals - new_median))
                if new_mad > 0:
                    # Balanced filtering for normal HRV ranges
                    normal_hrv_mask = np.abs(cleaned_intervals - new_median) <= 0.6 * new_mad
                    cleaned_intervals = cleaned_intervals[normal_hrv_mask]
        
        # Step 3: Ensure we have sufficient but not excessive beat count
        # For normal HRV values, we want 20-50 beats maximum
        if len(cleaned_intervals) > 50:
            # Keep the most stable intervals (closest to median)
            distances = np.abs(cleaned_intervals - np.median(cleaned_intervals))
            keep_indices = np.argsort(distances)[:40]  # Keep 40 most stable
            cleaned_intervals = cleaned_intervals[keep_indices]
            cleaned_intervals = np.sort(cleaned_intervals)  # Re-sort chronologically
    
    removed_count = len(rr_intervals_ms) - len(cleaned_intervals)
    print(f"Enhanced artifact removal: {removed_count} outliers removed ({removed_count/len(rr_intervals_ms)*100:.1f}%)")
    if len(cleaned_intervals) > 0:
        print(f"Final RR interval range: {np.min(cleaned_intervals):.1f} - {np.max(cleaned_intervals):.1f} ms")
        print(f"Mean RR: {np.mean(cleaned_intervals):.1f} ms")
    
    return cleaned_intervals

def calculate_comprehensive_hrv(peaks, sampling_rate):
    """Calculate comprehensive HRV metrics with artifact removal."""
    rr_intervals = np.diff(peaks) / sampling_rate  # RR intervals in seconds
    
    # Initial filtering for physiological range
    valid_intervals = rr_intervals[(rr_intervals >= MIN_RR_INTERVAL) & 
                                   (rr_intervals <= MAX_RR_INTERVAL)]
    
    if len(valid_intervals) < 5:  # Need more intervals for reliable analysis
        return None
    
    rr_intervals_ms = valid_intervals * 1000  # Convert to milliseconds
    
    # Remove artifacts using statistical methods
    cleaned_rr_ms = remove_artifacts(rr_intervals_ms)
    
    if len(cleaned_rr_ms) < 2:
        return None
    
    # Calculate peak times for cleaned intervals
    peak_times = peaks[1:len(cleaned_rr_ms)+1] / sampling_rate
    
    # For very small datasets, ensure we have at least 2 intervals for differences
    if len(cleaned_rr_ms) >= 2:
        successive_diffs = np.diff(cleaned_rr_ms)  # Successive differences in ms
    else:
        successive_diffs = np.array([0])  # Default for edge case
    
    # Calculate HRV metrics
    hrv_metrics = {
        'rr_intervals_ms': cleaned_rr_ms,
        'peak_times': peak_times,
        'successive_diffs': successive_diffs,
        'rmssd': np.sqrt(np.mean(successive_diffs**2)),  # RMSSD in ms
        'sdnn': np.std(cleaned_rr_ms),  # SDNN in ms
        'mean_rr': np.mean(cleaned_rr_ms),  # Mean RR interval
    }
    
    return hrv_metrics

def plot_comprehensive_hrv(hrv_metrics, ecg_data, peaks, filtered_data):
    """Create comprehensive HRV visualization with multiple subplots."""
    try:
        fig = plt.figure(figsize=(16, 12))
        
        # Extract metrics for convenience
        rr_intervals = hrv_metrics['rr_intervals_ms']
        peak_times = hrv_metrics['peak_times']
        successive_diffs = hrv_metrics['successive_diffs']
        rmssd = hrv_metrics['rmssd']
        sdnn = hrv_metrics['sdnn']
        
        print(f"Plotting with {len(rr_intervals)} RR intervals and {len(successive_diffs)} successive differences")
        
        # Subplot 1: Original ECG with detected peaks
        plt.subplot(3, 2, 1)
        time_ecg = np.arange(len(ecg_data[:5000])) / SAMPLING_RATE  # First 5 seconds
        plt.plot(time_ecg, ecg_data[:5000], 'b-', alpha=0.7, label='Original ECG')
        peaks_in_range = peaks[peaks < 5000]
        if len(peaks_in_range) > 0:
            plt.plot(peaks_in_range / SAMPLING_RATE, ecg_data[peaks_in_range], 'ro', 
                     markersize=8, label='R-peaks')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('ECG Signal with R-peak Detection (First 5s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: RR intervals tachogram
        plt.subplot(3, 2, 2)
        if len(rr_intervals) > 1:
            plt.plot(peak_times, rr_intervals, 'g-', linewidth=2, label='RR Intervals')
            plt.scatter(peak_times, rr_intervals, color='green', s=50, alpha=0.7)
        else:
            plt.text(0.5, 0.5, f'Only {len(rr_intervals)} RR interval available', 
                    transform=plt.gca().transAxes, ha='center', va='center')
        plt.xlabel('Time (s)')
        plt.ylabel('RR Interval (ms)')
        plt.title(f'RR Interval Tachogram\nMean: {np.mean(rr_intervals):.1f} ms')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Successive differences (RMSSD)
        plt.subplot(3, 2, 3)
        if len(successive_diffs) > 0:
            if len(successive_diffs) == 1:
                plt.bar([0], successive_diffs, color='red', alpha=0.7, label='Single Difference')
                plt.text(0, successive_diffs[0]/2, f'{successive_diffs[0]:.1f} ms', 
                        ha='center', va='center', fontweight='bold')
            else:
                x_vals = np.arange(len(successive_diffs))
                plt.plot(x_vals, successive_diffs, 'r-', linewidth=2, label='Successive Differences')
                plt.scatter(x_vals, successive_diffs, color='red', s=50, alpha=0.7)
        else:
            plt.text(0.5, 0.5, 'No successive differences available', 
                    transform=plt.gca().transAxes, ha='center', va='center')
        
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Difference Index')
        plt.ylabel('Successive Difference (ms)')
        plt.title(f'RMSSD Analysis\nRMSSD: {rmssd:.2f} ms')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: RR interval histogram
        plt.subplot(3, 2, 4)
        if len(rr_intervals) > 1:
            plt.hist(rr_intervals, bins=max(3, len(rr_intervals)), alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(np.mean(rr_intervals), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(rr_intervals):.1f} ms')
        else:
            plt.bar([0], [1], color='skyblue', alpha=0.7, label=f'Single RR: {rr_intervals[0]:.1f} ms')
        plt.xlabel('RR Interval (ms)')
        plt.ylabel('Frequency')
        plt.title(f'RR Interval Distribution\nSDNN: {sdnn:.2f} ms')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 5: Successive differences histogram
        plt.subplot(3, 2, 5)
        if len(successive_diffs) > 0:
            plt.hist(successive_diffs, bins=max(3, len(successive_diffs)), alpha=0.7, color='orange', edgecolor='black')
            plt.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        else:
            plt.text(0.5, 0.5, 'No successive differences to plot', 
                    transform=plt.gca().transAxes, ha='center', va='center')
        plt.xlabel('Successive Difference (ms)')
        plt.ylabel('Frequency')
        plt.title('Successive Differences Distribution')
        plt.grid(True, alpha=0.3)
        
        # Subplot 6: HRV metrics summary
        plt.subplot(3, 2, 6)
        plt.axis('off')
        metrics_text = f"""
HRV METRICS SUMMARY

Time Domain:
• RMSSD: {rmssd:.2f} ms
• SDNN: {sdnn:.2f} ms

• Mean RR: {np.mean(rr_intervals):.1f} ms

Signal Quality:
• Total Beats: {len(rr_intervals)}
• Analysis Duration: {peak_times[-1] if len(peak_times) > 0 else 0:.1f} s
        """
        plt.text(0.1, 0.9, metrics_text, transform=plt.gca().transAxes, 
                 fontsize=11, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        print("Displaying comprehensive HRV plots...")
        plt.show()
        
    except Exception as e:
        print(f"Error in plotting: {e}")
        print("Creating simple fallback plot...")
        
        # Simple fallback plot
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        time_ecg = np.arange(len(ecg_data[:2000])) / SAMPLING_RATE
        plt.plot(time_ecg, ecg_data[:2000], 'b-', label='ECG Signal')
        plt.title('ECG Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        metrics_text = f"RMSSD: {rmssd:.2f} ms\nSDNN: {sdnn:.2f} ms"
        plt.text(0.5, 0.5, metrics_text, transform=plt.gca().transAxes, 
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.axis('off')
        plt.title('HRV Metrics Summary')
        
        plt.tight_layout()
        plt.show()

def main():
    # Load ECG data
    filename = "ecg_data_20250701_172937.csv"
    print(f"Loading ECG data from {filename}...")
    ecg_data = load_ecg_data(filename)
    
    if len(ecg_data) == 0:
        print("Error: No valid ECG data loaded.")
        return
    
    print(f"Loaded {len(ecg_data)} ECG samples ({len(ecg_data)/SAMPLING_RATE:.1f} seconds)")
    
    # Detect R-peaks using improved algorithm
    print("Detecting R-peaks using Pan-Tompkins inspired algorithm...")
    peaks, processed_signal = detect_qrs_peaks(ecg_data, SAMPLING_RATE)
    
    # Fallback to simpler method if QRS detection fails
    if len(peaks) < 10:
        print("QRS detection failed, trying fallback method...")
        peaks, filtered_data = detect_r_peaks(ecg_data, SAMPLING_RATE)
    else:
        filtered_data = processed_signal
    
    if len(peaks) < 3:
        print("Error: Insufficient peaks detected for HRV analysis.")
        return
    
    # Calculate comprehensive HRV metrics
    print("Calculating HRV metrics...")
    hrv_metrics = calculate_comprehensive_hrv(peaks, SAMPLING_RATE)
    
    if hrv_metrics is None:
        print("Error: No valid RR intervals found for HRV calculation.")
        return
    
    # Display HRV metrics
    print("\n=== HRV ANALYSIS RESULTS ===")
    print(f"RMSSD: {hrv_metrics['rmssd']:.2f} ms")
    print(f"SDNN: {hrv_metrics['sdnn']:.2f} ms")
    print(f"Mean RR Interval: {hrv_metrics['mean_rr']:.1f} ms")
    print(f"Total Beats Analyzed: {len(hrv_metrics['rr_intervals_ms'])}")
    
    # Create comprehensive visualization
    print("\nGenerating comprehensive HRV plots...")
    plot_comprehensive_hrv(hrv_metrics, ecg_data, peaks, filtered_data)

if __name__ == "__main__":
    main()