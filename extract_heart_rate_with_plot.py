import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import csv

# Constants
SAMPLING_RATE = 1000  # Hz (samples per second), adjust if known
MIN_RR_INTERVAL = 0.3  # Minimum RR interval in seconds (200 BPM max)
MAX_RR_INTERVAL = 2.0  # Maximum RR interval in seconds (30 BPM min)

def load_ecg_data(filename):
    """Load ECG amplitude data from a CSV file."""
    ecg_data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if row:  # Ensure row is not empty
                try:
                    ecg_data.append(float(row[0]))
                except ValueError:
                    continue  # Skip invalid data
    return np.array(ecg_data)

def detect_r_peaks(ecg_data, sampling_rate):
    """Detect R-peaks in the ECG signal."""
    # Use a height threshold (e.g., 80% of max amplitude) to detect prominent peaks
    height_threshold = np.max(ecg_data) * 0.8
    min_distance = int(MIN_RR_INTERVAL * sampling_rate)  # Minimum distance between peaks
    
    # Find peaks using scipy's find_peaks
    peaks, _ = find_peaks(ecg_data, height=height_threshold, distance=min_distance)
    
    return peaks

def calculate_heart_rate(peaks, sampling_rate):
    """Calculate heart rate from RR intervals."""
    # Calculate RR intervals (in samples)
    rr_intervals = np.diff(peaks)
    
    # Convert to seconds
    rr_intervals_sec = rr_intervals / sampling_rate
    
    # Filter out invalid intervals (outside realistic BPM range)
    valid_intervals = rr_intervals_sec[(rr_intervals_sec >= MIN_RR_INTERVAL) & 
                                      (rr_intervals_sec <= MAX_RR_INTERVAL)]
    
    if len(valid_intervals) == 0:
        return None  # No valid intervals found
    
    # Calculate heart rate (BPM) = 60 / average RR interval (in seconds)
    avg_rr_interval = np.mean(valid_intervals)
    heart_rate = 60 / avg_rr_interval
    
    return heart_rate

def plot_ecg_with_peaks(ecg_data, peaks, sampling_rate, heart_rate):
    """Plot ECG signal with detected R-peaks and heart rate."""
    # Create time array (in seconds)
    time = np.arange(len(ecg_data)) / sampling_rate
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(time, ecg_data, label='ECG Signal', color='blue')
    plt.plot(time[peaks], ecg_data[peaks], 'ro', label='R-Peaks', markersize=8)
    
    # Add labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'ECG Signal with R-Peaks\nEstimated Heart Rate: {heart_rate:.2f} BPM' if heart_rate else 'ECG Signal with R-Peaks\nNo Valid Heart Rate Calculated')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

def main():
    # Load ECG data
    filename = "ecg_data_20250701_172937.csv"
    ecg_data = load_ecg_data(filename)
    
    if len(ecg_data) == 0:
        print("Error: No valid ECG data loaded.")
        return
    
    # Detect R-peaks
    peaks = detect_r_peaks(ecg_data, SAMPLING_RATE)
    
    if len(peaks) < 2:
        print("Error: Insufficient peaks detected for heart rate calculation.")
        heart_rate = None
    else:
        # Calculate heart rate
        heart_rate = calculate_heart_rate(peaks, SAMPLING_RATE)
    
    # Plot the ECG signal with detected peaks
    plot_ecg_with_peaks(ecg_data, peaks, SAMPLING_RATE, heart_rate)
    
    if heart_rate is None:
        print("Error: No valid RR intervals found for heart rate calculation.")
    else:
        print(f"Estimated Heart Rate: {heart_rate:.2f} BPM")

if __name__ == "__main__":
    main()
