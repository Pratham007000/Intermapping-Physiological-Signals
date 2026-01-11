import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def generate_realistic_ecg(duration=30, sampling_rate=500, heart_rate=75):
    """Generate realistic ECG signal with proper morphology."""
    
    # Time array
    t = np.linspace(0, duration, int(duration * sampling_rate))
    
    # Heart rate variability (small random variations)
    rr_intervals = 60 / heart_rate + np.random.normal(0, 0.02, len(t))
    rr_intervals = np.clip(rr_intervals, 0.5, 1.5)  # Physiological limits
    
    # Generate ECG signal
    ecg_signal = np.zeros_like(t)
    
    # Add baseline noise
    baseline_noise = np.random.normal(0, 0.05, len(t))
    
    # Generate R-peaks and complete PQRST complexes
    current_time = 0
    peak_times = []
    
    i = 0
    while current_time < duration - 1:
        if i < len(rr_intervals):
            rr_interval = rr_intervals[i]
        else:
            rr_interval = 60 / heart_rate
        
        peak_time = current_time + rr_interval
        if peak_time < duration:
            peak_times.append(peak_time)
        
        # Add PQRST complex
        peak_idx = int(peak_time * sampling_rate)
        
        # P wave (small positive deflection)
        p_start = peak_idx - int(0.15 * sampling_rate)
        p_end = peak_idx - int(0.05 * sampling_rate)
        if p_start >= 0 and p_end < len(ecg_signal):
            p_wave = 0.2 * np.exp(-((np.arange(p_start, p_end) - (p_start + p_end)//2)**2) / (0.02 * sampling_rate)**2)
            ecg_signal[p_start:p_end] += p_wave
        
        # QRS complex (main spike)
        qrs_start = peak_idx - int(0.04 * sampling_rate)
        qrs_end = peak_idx + int(0.06 * sampling_rate)
        if qrs_start >= 0 and qrs_end < len(ecg_signal):
            # Q wave (small negative)
            q_idx = qrs_start + int(0.01 * sampling_rate)
            if q_idx < len(ecg_signal):
                ecg_signal[q_idx] -= 0.3
            
            # R wave (large positive)
            if peak_idx < len(ecg_signal):
                ecg_signal[peak_idx] += 1.0 + np.random.normal(0, 0.1)
            
            # S wave (negative after R)
            s_idx = peak_idx + int(0.02 * sampling_rate)
            if s_idx < len(ecg_signal):
                ecg_signal[s_idx] -= 0.4
        
        # T wave (positive deflection after QRS)
        t_start = peak_idx + int(0.1 * sampling_rate)
        t_end = peak_idx + int(0.3 * sampling_rate)
        if t_start >= 0 and t_end < len(ecg_signal):
            t_wave = 0.3 * np.exp(-((np.arange(t_start, t_end) - (t_start + t_end)//2)**2) / (0.08 * sampling_rate)**2)
            ecg_signal[t_start:t_end] += t_wave
        
        current_time = peak_time
        i += 1
    
    # Add baseline wander (low frequency drift)
    baseline_freq = 0.5  # Hz
    baseline_wander = 0.1 * np.sin(2 * np.pi * baseline_freq * t)
    
    # Add power line interference (50/60 Hz)
    powerline_freq = 60  # Hz
    powerline_noise = 0.03 * np.sin(2 * np.pi * powerline_freq * t)
    
    # Combine all components
    ecg_signal = ecg_signal + baseline_noise + baseline_wander + powerline_noise
    
    # Apply smoothing filter
    nyquist = sampling_rate / 2
    low_cutoff = 0.5 / nyquist
    high_cutoff = 40 / nyquist
    b, a = butter(4, [low_cutoff, high_cutoff], btype='band')
    ecg_signal = filtfilt(b, a, ecg_signal)
    
    # Normalize to realistic ECG amplitude range (0.5-2.0 mV)
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
    ecg_signal = ecg_signal * 0.5 + 1.0  # Scale to 0.5-1.5 mV range
    
    return ecg_signal, peak_times, sampling_rate

def generate_realistic_ppg(ecg_signal, peak_times, sampling_rate, delay_ms=200):
    """Generate realistic PPG signal from ECG with physiological delay."""
    
    ppg_signal = np.zeros_like(ecg_signal)
    delay_samples = int(delay_ms * sampling_rate / 1000)
    
    for peak_time in peak_times:
        peak_idx = int(peak_time * sampling_rate) + delay_samples
        
        if peak_idx < len(ppg_signal):
            # Create realistic PPG pulse morphology
            pulse_width = int(0.4 * sampling_rate)  # 400ms pulse width
            
            # Systolic peak (sharp rise, slower fall)
            rise_time = pulse_width // 4
            fall_time = pulse_width * 3 // 4
            
            # Rising edge (steep)
            start_rise = max(0, peak_idx - rise_time//2)
            end_rise = peak_idx + rise_time//2
            if start_rise < len(ppg_signal) and end_rise <= len(ppg_signal):
                rise_indices = np.arange(start_rise, end_rise)
                rise_values = np.exp(4 * (rise_indices - start_rise) / rise_time - 4)
                ppg_signal[start_rise:end_rise] += rise_values * 0.8
            
            # Falling edge with dicrotic notch
            start_fall = peak_idx
            end_fall = min(len(ppg_signal), peak_idx + fall_time)
            if start_fall < len(ppg_signal) and end_fall > start_fall:
                fall_indices = np.arange(start_fall, end_fall)
                
                # Primary fall
                fall_values = np.exp(-2 * (fall_indices - start_fall) / fall_time)
                
                # Add dicrotic notch (small secondary peak)
                dicrotic_time = start_fall + fall_time // 3
                if dicrotic_time < len(ppg_signal):
                    dicrotic_width = fall_time // 8
                    dicrotic_indices = np.arange(max(start_fall, dicrotic_time - dicrotic_width//2),
                                                min(end_fall, dicrotic_time + dicrotic_width//2))
                    if len(dicrotic_indices) > 0:
                        dicrotic_values = 0.3 * np.exp(-((dicrotic_indices - dicrotic_time)**2) / (dicrotic_width/4)**2)
                        fall_values[dicrotic_indices - start_fall] += dicrotic_values
                
                ppg_signal[start_fall:end_fall] += fall_values * 0.8
    
    # Add realistic PPG noise and artifacts
    # Respiratory modulation (breathing effect)
    t = np.arange(len(ppg_signal)) / sampling_rate
    resp_freq = 0.25  # 15 breaths per minute
    respiratory_modulation = 0.1 * np.sin(2 * np.pi * resp_freq * t)
    
    # Motion artifacts (occasional spikes)
    motion_artifacts = np.zeros_like(ppg_signal)
    num_artifacts = np.random.poisson(3)  # Average 3 artifacts
    for _ in range(num_artifacts):
        artifact_time = np.random.uniform(0, len(ppg_signal))
        artifact_idx = int(artifact_time)
        if artifact_idx < len(motion_artifacts):
            artifact_magnitude = np.random.uniform(0.2, 0.8)
            artifact_width = np.random.randint(5, 20)
            start_idx = max(0, artifact_idx - artifact_width//2)
            end_idx = min(len(motion_artifacts), artifact_idx + artifact_width//2)
            motion_artifacts[start_idx:end_idx] += artifact_magnitude
    
    # Combine PPG components
    ppg_signal = ppg_signal + respiratory_modulation + motion_artifacts
    
    # Add measurement noise
    measurement_noise = np.random.normal(0, 0.02, len(ppg_signal))
    ppg_signal += measurement_noise
    
    # Normalize PPG signal
    ppg_signal = (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)
    
    return ppg_signal

def main():
    """Generate and save realistic ECG and PPG data."""
    
    print("Generating realistic ECG and PPG data...")
    
    # Generate ECG
    duration = 30  # seconds
    sampling_rate = 500  # Hz
    heart_rate = 75  # BPM
    
    ecg_signal, peak_times, fs = generate_realistic_ecg(duration, sampling_rate, heart_rate)
    
    # Generate corresponding PPG
    ppg_signal = generate_realistic_ppg(ecg_signal, peak_times, sampling_rate, delay_ms=200)
    
    # Create DataFrame
    data = {
        'ECG Amplitude': ecg_signal,
        'PPG Amplitude': ppg_signal
    }
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('realistic_ecg_ppg_data.csv', index=False)
    print(f"✓ Saved {len(ecg_signal)} samples to 'realistic_ecg_ppg_data.csv'")
    print(f"Duration: {duration}s, Sampling Rate: {sampling_rate}Hz")
    print(f"Heart Rate: {heart_rate} BPM, R-peaks: {len(peak_times)}")
    
    # Quick visualization
    plt.figure(figsize=(12, 8))
    
    # Show first 5 seconds
    samples_to_show = 5 * sampling_rate
    time_axis = np.arange(samples_to_show) / sampling_rate
    
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, ecg_signal[:samples_to_show], 'b-', linewidth=1.5, label='ECG')
    plt.title('Generated Realistic ECG Signal (First 5 seconds)')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, ppg_signal[:samples_to_show], 'r-', linewidth=1.5, label='PPG')
    plt.title('Generated Realistic PPG Signal (First 5 seconds)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
