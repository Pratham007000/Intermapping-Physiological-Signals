import pyedflib
import pandas as pd
import numpy as np
import neurokit2 as nk
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import signal as scipy_signal
from scipy.stats import pearsonr, spearmanr
import joblib
import os
import warnings
import logging
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class Config:
    """Configuration class for the EMG prediction pipeline"""
    data_dir: str = './physionet.org/files/scientisst-move-biosignals/1.0.1'
    min_signal_length_seconds: int = 10  # Minimum signal length in seconds
    max_signal_length_seconds: int = 300  # Maximum signal length to process
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_jobs: int = -1

# Step 1: Load ScientISST MOVE dataset from .edf files
def load_scientisst_move_data(chest_path, forearm_path):
    # Load chest data (contains S1, S2 ECG)
    with pyedflib.EdfReader(chest_path) as chest_file:
        chest_signals = []
        chest_signal_names = chest_file.getSignalLabels()
        fs = chest_file.getSampleFrequency(0)  # Assuming same sampling rate
        
        # Read all signals from chest
        for i in range(chest_file.signals_in_file):
            chest_signals.append(chest_file.readSignal(i))
        
        # Find ECG channels (looking for ecg:dry, ecg:gel, S1, S2, or ECG)
        ecg_idx = None
        for i, name in enumerate(chest_signal_names):
            name_lower = name.lower()
            if ('ecg' in name_lower) or ('s1' in name_lower) or ('s2' in name_lower):
                ecg_idx = i
                break
        
        if ecg_idx is None:
            raise ValueError(f"ECG signal not found in chest data. Available signals: {chest_signal_names}")
            
        ecg_signal = chest_signals[ecg_idx]
    
    # Load forearm data (contains S9 EMG)
    with pyedflib.EdfReader(forearm_path) as forearm_file:
        forearm_signals = []
        forearm_signal_names = forearm_file.getSignalLabels()
        
        # Read all signals from forearm
        for i in range(forearm_file.signals_in_file):
            forearm_signals.append(forearm_file.readSignal(i))
        
        # Find EMG channel (looking for emg, S9, or EMG)
        emg_idx = None
        for i, name in enumerate(forearm_signal_names):
            name_lower = name.lower()
            if ('emg' in name_lower) or ('s9' in name_lower):
                emg_idx = i
                break
        
        if emg_idx is None:
            raise ValueError(f"EMG signal not found in forearm data. Available signals: {forearm_signal_names}")
        
        emg_signal = forearm_signals[emg_idx]
    
    return ecg_signal, emg_signal, int(fs)

# Step 2: Preprocess and extract features
def extract_features(ecg_signal, emg_signal, fs):
    # Validate minimum signal lengths before processing
    min_samples_for_ecg = max(100, fs * 2)  # At least 2 seconds or 100 samples, whichever is larger
    min_samples_for_emg = max(50, fs * 1)   # At least 1 second or 50 samples, whichever is larger
    
    if len(ecg_signal) < min_samples_for_ecg:
        raise ValueError(f"ECG signal too short: {len(ecg_signal)} samples (minimum {min_samples_for_ecg} required)")
    
    if len(emg_signal) < min_samples_for_emg:
        raise ValueError(f"EMG signal too short: {len(emg_signal)} samples (minimum {min_samples_for_emg} required)")
    
    # Additional validation for sampling rate
    if fs < 50:  # Minimum reasonable sampling rate for physiological signals
        raise ValueError(f"Sampling rate too low: {fs} Hz (minimum 50 Hz required)")
    
    # Try processing ECG with multiple approaches for robustness
    ecg_processed = None
    ecg_info = None
    
    # First, try with a smaller chunk if the signal is very long
    if len(ecg_signal) > fs * 300:  # If longer than 5 minutes, use first 5 minutes
        ecg_chunk = ecg_signal[:fs * 300]
    else:
        ecg_chunk = ecg_signal
    
    # Additional check after chunking
    if len(ecg_chunk) < min_samples_for_ecg:
        raise ValueError(f"ECG chunk too short after processing: {len(ecg_chunk)} samples")
    
    # Try multiple ECG processing methods with progressively simpler approaches
    ecg_processing_successful = False
    
    # Method 1: Try standard NeuroKit2 processing
    try:
        ecg_processed, ecg_info = nk.ecg_process(ecg_chunk, sampling_rate=fs)
        ecg_processing_successful = True
    except (IndexError, ValueError) as e:
        print(f"    Standard ECG processing failed: {e}")
        
        # Method 2: Try step-by-step processing with error handling
        try:
            # Clean the signal first
            cleaned_ecg = nk.ecg_clean(ecg_chunk, sampling_rate=fs, method='pantompkins1985')
            
            # Detect R-peaks with error handling
            try:
                _, r_peaks_dict = nk.ecg_peaks(cleaned_ecg, sampling_rate=fs, method='pantompkins1985')
                r_peaks = r_peaks_dict['ECG_R_Peaks']
            except (IndexError, ValueError):
                # Try with different method
                _, r_peaks_dict = nk.ecg_peaks(cleaned_ecg, sampling_rate=fs, method='hamilton2002')
                r_peaks = r_peaks_dict['ECG_R_Peaks']
            
            ecg_info = {'ECG_R_Peaks': r_peaks}
            ecg_processed = {'ECG_Clean': cleaned_ecg}
            ecg_processing_successful = True
            
        except (IndexError, ValueError) as e:
            print(f"    Alternative ECG processing failed: {e}")
            
            # Method 3: Basic peak detection using scipy
            try:
                
                # Basic preprocessing
                cleaned_ecg = ecg_chunk - np.mean(ecg_chunk)
                if np.std(cleaned_ecg) > 0:
                    cleaned_ecg = cleaned_ecg / np.std(cleaned_ecg)
                else:
                    raise ValueError("ECG signal has zero variance")
                
                # Find peaks with adaptive parameters
                min_distance = max(int(fs * 0.4), 1)  # Minimum 0.4s between R-peaks
                peaks, _ = scipy_signal.find_peaks(cleaned_ecg, 
                                                 height=np.mean(cleaned_ecg) + 0.5 * np.std(cleaned_ecg),
                                                 distance=min_distance)
                
                if len(peaks) == 0:
                    # Try with lower threshold
                    peaks, _ = scipy_signal.find_peaks(cleaned_ecg, 
                                                     height=np.mean(cleaned_ecg),
                                                     distance=min_distance)
                
                if len(peaks) == 0:
                    raise ValueError("No peaks detected with basic method")
                    
                ecg_info = {'ECG_R_Peaks': peaks}
                ecg_processed = {'ECG_Clean': cleaned_ecg}
                ecg_processing_successful = True
                
            except Exception as e:
                raise ValueError(f"All ECG processing methods failed: {e}")
    
    if not ecg_processing_successful:
        raise ValueError("ECG processing completely failed")
    
    # Check if R-peaks were detected and are valid
    r_peaks = ecg_info['ECG_R_Peaks']
    if len(r_peaks) == 0:
        raise ValueError("No R-peaks detected in ECG signal")
    
    # Filter R-peaks to ensure they are within bounds of the signal
    valid_r_peaks = r_peaks[r_peaks < len(ecg_signal)]
    if len(valid_r_peaks) == 0:
        raise ValueError("No valid R-peaks within signal bounds")
    
    # Calculate heart rate with more robust error handling
    try:
        hr = nk.ecg_rate(valid_r_peaks, sampling_rate=fs, desired_length=len(ecg_signal))
        
        # Handle case where HR calculation fails or returns empty array
        if len(hr) == 0 or np.all(np.isnan(hr)):
            raise ValueError("HR calculation returned empty or NaN values")
            
    except (IndexError, ValueError) as e:
        # Fallback: estimate HR from R-peak intervals
        if len(valid_r_peaks) > 1:
            rr_intervals = np.diff(valid_r_peaks) / fs  # RR intervals in seconds
            if len(rr_intervals) > 0 and np.mean(rr_intervals) > 0:
                hr_estimate = 60.0 / np.mean(rr_intervals)  # Convert to BPM
                # Ensure HR is within reasonable physiological range (30-200 BPM)
                hr_estimate = np.clip(hr_estimate, 30, 200)
                hr = np.full(len(ecg_signal), hr_estimate)
            else:
                raise ValueError("Invalid RR intervals calculated")
        else:
            raise ValueError("Insufficient R-peaks for heart rate calculation")
    
    # Process EMG signal
    try:
        emg_processed, emg_info = nk.emg_process(emg_signal, sampling_rate=fs)
        
        # Handle different ways NeuroKit2 might return EMG amplitude data
        emg_amplitude = None
        if isinstance(emg_processed, dict):
            emg_amplitude = emg_processed.get('EMG_Amplitude', None)
        elif hasattr(emg_processed, 'EMG_Amplitude'):
            try:
                emg_amplitude = emg_processed['EMG_Amplitude'].values if hasattr(emg_processed['EMG_Amplitude'], 'values') else emg_processed['EMG_Amplitude']
            except:
                emg_amplitude = None
        
        # Fallback: calculate simple EMG amplitude if NeuroKit processing fails
        if emg_amplitude is None or len(emg_amplitude) == 0:
            # Simple envelope calculation as fallback
            emg_amplitude = np.abs(emg_signal)
            
    except (IndexError, ValueError) as e:
        # Fallback: calculate simple EMG amplitude
        print(f"    EMG processing failed, using simple amplitude calculation: {e}")
        emg_amplitude = np.abs(emg_signal)
    
    # Ensure EMG amplitude is valid
    if len(emg_amplitude) == 0 or np.all(np.isnan(emg_amplitude)):
        raise ValueError("EMG amplitude calculation failed")
    
    # Extract features with additional validation
    hr_valid = hr[~np.isnan(hr)]
    if len(hr_valid) == 0:
        raise ValueError("All HR values are NaN")
    
    emg_valid = emg_amplitude[~np.isnan(emg_amplitude)]
    if len(emg_valid) == 0:
        raise ValueError("All EMG amplitude values are NaN")
    
    # Enhanced ECG features with advanced HRV analysis
    rr_intervals = np.diff(valid_r_peaks) / fs  # RR intervals in seconds
    hrv_features = {}
    
    if len(rr_intervals) > 3:  # Need at least 4 R-peaks for meaningful HRV
        # Time domain HRV features
        nn_intervals = rr_intervals * 1000  # Convert to milliseconds
        diff_nn = np.diff(nn_intervals)
        
        hrv_features = {
            'rmssd': np.sqrt(np.mean(diff_nn ** 2)),  # RMSSD in ms
            'sdnn': np.std(nn_intervals),  # SDNN in ms
            'pnn50': np.sum(np.abs(diff_nn) > 50) / len(diff_nn) * 100,  # pNN50
            'pnn20': np.sum(np.abs(diff_nn) > 20) / len(diff_nn) * 100,  # pNN20
            'cvnn': np.std(nn_intervals) / np.mean(nn_intervals) * 100,  # Coefficient of variation
            'triangular_index': len(nn_intervals) / np.max(np.histogram(nn_intervals, bins=50)[0]),
            'tinn': np.ptp(nn_intervals),  # Triangular interpolation of NN interval histogram
            'hr_variability_range': np.ptp(60000 / nn_intervals)  # HR variability range
        }
        
        # Frequency domain features (if enough data)
        if len(nn_intervals) > fs * 2:  # At least 2 seconds of RR data
            try:
                # Interpolate RR intervals for frequency analysis
                time_rr = np.cumsum(nn_intervals) / 1000  # Time in seconds
                interp_time = np.arange(0, time_rr[-1], 1/4)  # 4 Hz interpolation
                interp_rr = np.interp(interp_time, time_rr, nn_intervals)
                
                # Power spectral density
                freqs, psd_hrv = scipy_signal.welch(interp_rr, fs=4, nperseg=min(len(interp_rr)//2, 256))
                
                # HRV frequency bands
                vlf_band = (freqs >= 0.003) & (freqs < 0.04)  # Very low frequency
                lf_band = (freqs >= 0.04) & (freqs < 0.15)    # Low frequency  
                hf_band = (freqs >= 0.15) & (freqs < 0.4)     # High frequency
                
                vlf_power = np.trapezoid(psd_hrv[vlf_band], freqs[vlf_band]) if np.any(vlf_band) else 0
                lf_power = np.trapezoid(psd_hrv[lf_band], freqs[lf_band]) if np.any(lf_band) else 0
                hf_power = np.trapezoid(psd_hrv[hf_band], freqs[hf_band]) if np.any(hf_band) else 0
                
                total_power_hrv = vlf_power + lf_power + hf_power
                
                hrv_features.update({
                    'vlf_power': vlf_power,
                    'lf_power': lf_power, 
                    'hf_power': hf_power,
                    'lf_hf_ratio': lf_power / hf_power if hf_power > 0 else 0,
                    'total_power_hrv': total_power_hrv
                })
            except Exception as e:
                # Fallback values if frequency analysis fails
                print(f"    HRV frequency analysis failed: {e}")
                hrv_features.update({
                    'vlf_power': 0, 'lf_power': 0, 'hf_power': 0, 
                    'lf_hf_ratio': 0, 'total_power_hrv': 0
                })
        else:
            # Not enough data for frequency analysis
            hrv_features.update({
                'vlf_power': 0, 'lf_power': 0, 'hf_power': 0, 
                'lf_hf_ratio': 0, 'total_power_hrv': 0
            })
    else:
        # Default values when insufficient R-peaks
        hrv_features = {
            'rmssd': 0, 'sdnn': 0, 'pnn50': 0, 'pnn20': 0, 'cvnn': 0,
            'triangular_index': 0, 'tinn': 0, 'hr_variability_range': 0,
            'vlf_power': 0, 'lf_power': 0, 'hf_power': 0, 
            'lf_hf_ratio': 0, 'total_power_hrv': 0
        }
    
    # ECG frequency domain features
    # Handle different ways NeuroKit2 might return processed data
    ecg_clean = None
    if ecg_processed is not None:
        if isinstance(ecg_processed, dict):
            ecg_clean = ecg_processed.get('ECG_Clean', None)
        elif hasattr(ecg_processed, 'ECG_Clean'):
            try:
                ecg_clean = ecg_processed['ECG_Clean'].values if hasattr(ecg_processed['ECG_Clean'], 'values') else ecg_processed['ECG_Clean']
            except:
                ecg_clean = None
    
    # Fallback to original signal if processed version is not available
    if ecg_clean is None or len(ecg_clean) == 0:
        ecg_clean = ecg_chunk
    
    if len(ecg_clean) > fs:
        try:
            freqs, psd = scipy_signal.welch(ecg_clean, fs, nperseg=min(len(ecg_clean)//4, fs*4))
            # Find dominant frequency in ECG range (0.5-40 Hz)
            ecg_freq_mask = (freqs >= 0.5) & (freqs <= 40)
            if np.any(ecg_freq_mask):
                dominant_freq = freqs[ecg_freq_mask][np.argmax(psd[ecg_freq_mask])]
                total_power = np.trapezoid(psd[ecg_freq_mask], freqs[ecg_freq_mask])
            else:
                dominant_freq = 0
                total_power = 0
        except Exception:
            dominant_freq = 0
            total_power = 0
    else:
        dominant_freq = 0
        total_power = 0
    
    # EMG frequency domain features
    if len(emg_signal) > fs:
        freqs_emg, psd_emg = scipy_signal.welch(emg_signal, fs, nperseg=min(len(emg_signal)//4, fs*4))
        # EMG typically has energy in 10-500 Hz range
        emg_freq_mask = (freqs_emg >= 10) & (freqs_emg <= 500)
        if np.any(emg_freq_mask):
            emg_total_power = np.trapezoid(psd_emg[emg_freq_mask], freqs_emg[emg_freq_mask])
            emg_median_freq = freqs_emg[emg_freq_mask][np.argmin(np.abs(np.cumsum(psd_emg[emg_freq_mask]) - np.sum(psd_emg[emg_freq_mask])/2))]
        else:
            emg_total_power = 0
            emg_median_freq = 0
    else:
        emg_total_power = 0
        emg_median_freq = 0
    
    # Advanced EMG features
    emg_rms = np.sqrt(np.mean(emg_valid ** 2))  # RMS
    emg_var = np.var(emg_valid)  # Variance
    
    # Calculate skewness and kurtosis using scipy.stats
    from scipy.stats import skew, kurtosis
    emg_skewness = skew(emg_valid) if len(emg_valid) > 2 else 0
    emg_kurtosis = kurtosis(emg_valid) if len(emg_valid) > 3 else 0
    
    # Zero crossing rate for EMG
    zero_crossings = np.sum(np.diff(np.sign(emg_valid - np.mean(emg_valid))) != 0)
    emg_zcr = zero_crossings / len(emg_valid) if len(emg_valid) > 1 else 0
    
    # Combine all ECG features (expanded)
    ecg_features = {
        'hr_mean': np.mean(hr_valid),
        'hr_std': np.std(hr_valid),
        'hr_max': np.max(hr_valid),
        'hr_min': np.min(hr_valid),
        'hr_range': np.ptp(hr_valid),
        'hr_cv': np.std(hr_valid) / np.mean(hr_valid) if np.mean(hr_valid) > 0 else 0,
        'rmssd': hrv_features['rmssd'],
        'sdnn': hrv_features['sdnn'],
        'pnn50': hrv_features['pnn50'],
        'pnn20': hrv_features['pnn20'],
        'cvnn': hrv_features['cvnn'],
        'triangular_index': hrv_features['triangular_index'],
        'tinn': hrv_features['tinn'],
        'hr_variability_range': hrv_features['hr_variability_range'],
        'vlf_power': hrv_features['vlf_power'],
        'lf_power': hrv_features['lf_power'],
        'hf_power': hrv_features['hf_power'],
        'lf_hf_ratio': hrv_features['lf_hf_ratio'],
        'total_power_hrv': hrv_features['total_power_hrv'],
        'ecg_dominant_freq': dominant_freq,
        'ecg_total_power': total_power,
        'num_rpeaks': len(valid_r_peaks),
        'rr_mean': np.mean(rr_intervals) * 1000 if len(rr_intervals) > 0 else 0,
        'rr_std': np.std(rr_intervals) * 1000 if len(rr_intervals) > 0 else 0
    }
    
    # Combine all EMG features (expanded)
    emg_features = {
        'emg_amplitude_mean': np.mean(emg_valid),
        'emg_amplitude_std': np.std(emg_valid),
        'emg_amplitude_max': np.max(emg_valid),
        'emg_amplitude_min': np.min(emg_valid),
        'emg_amplitude_range': np.ptp(emg_valid),
        'emg_rms': emg_rms,
        'emg_variance': emg_var,
        'emg_skewness': emg_skewness,
        'emg_kurtosis': emg_kurtosis,
        'emg_zcr': emg_zcr,
        'emg_total_power': emg_total_power,
        'emg_median_freq': emg_median_freq
    }
    
    return ecg_features, emg_features

# Step 3: Main script
def main():
    # Specify the dataset directory
    data_dir = './physionet.org/files/scientisst-move-biosignals/1.0.1'
    
    # Find all record subfolders
    records = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Initialize feature lists
    ecg_feature_list = []
    emg_feature_list = []
    
    # Process each record
    for record in records:
        chest_file = os.path.join(record, 'scientisst_chest.edf')
        forearm_file = os.path.join(record, 'scientisst_forearm.edf')
        
        if not os.path.exists(chest_file) or not os.path.exists(forearm_file):
            print(f"Missing chest or forearm file in {record}")
            continue
        
        try:
            ecg_signal, emg_signal, fs = load_scientisst_move_data(chest_file, forearm_file)
            
            # Validate signal lengths
            if len(ecg_signal) == 0 or len(emg_signal) == 0:
                print(f"Error processing {record}: Empty signals detected")
                continue
            
            # Ensure signals have sufficient length for processing
            min_length = fs * 5  # At least 5 seconds of data
            if len(ecg_signal) < min_length or len(emg_signal) < min_length:
                print(f"Error processing {record}: Signals too short (min {min_length} samples required)")
                continue
            
            ecg_feats, emg_feats = extract_features(ecg_signal, emg_signal, fs)
            
            ecg_feature_list.append(list(ecg_feats.values()))
            emg_feature_list.append(list(emg_feats.values()))
            
        except ValueError as e:
            print(f"Error processing {record}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error processing {record}: {e}")
            continue
    
    # Convert to DataFrame with all expanded features
    ecg_df = pd.DataFrame(ecg_feature_list, 
                         columns=['hr_mean', 'hr_std', 'hr_max', 'hr_min', 'hr_range', 'hr_cv',
                                  'rmssd', 'sdnn', 'pnn50', 'pnn20', 'cvnn', 'triangular_index', 'tinn',
                                  'hr_variability_range', 'vlf_power', 'lf_power', 'hf_power', 'lf_hf_ratio',
                                  'total_power_hrv', 'ecg_dominant_freq', 'ecg_total_power', 'num_rpeaks',
                                  'rr_mean', 'rr_std'])
    emg_df = pd.DataFrame(emg_feature_list, 
                         columns=['emg_amplitude_mean', 'emg_amplitude_std', 'emg_amplitude_max',
                                  'emg_amplitude_min', 'emg_amplitude_range', 'emg_rms', 'emg_variance', 
                                  'emg_skewness', 'emg_kurtosis', 'emg_zcr', 'emg_total_power', 'emg_median_freq'])
    
    # Prepare data for modeling
    X = ecg_df.values
    y = emg_df.values
    
    if len(X) == 0 or len(y) == 0:
        print("No valid data processed. Check directory structure or signal availability.")
        return
    
    print(f"Processed {len(X)} valid records")
    print(f"ECG features shape: {X.shape}")
    print(f"EMG features shape: {y.shape}")
    
    # Check for NaN or infinite values
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X_clean)
    y_scaled = scaler_y.fit_transform(y_clean)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # Define multiple models to compare (with multi-output support)
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ), n_jobs=-1),
        'Ridge Regression': Ridge(
            alpha=1.0,
            random_state=42
        ),
        'ElasticNet': ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            random_state=42,
            max_iter=2000
        )
    }
    
    print("\n=== MODEL COMPARISON ===")
    best_model = None
    best_score = float('-inf')
    best_name = ""
    
    model_results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        
        # Transform back to original scale for evaluation
        y_test_orig = scaler_y.inverse_transform(y_test)
        y_pred_orig = scaler_y.inverse_transform(y_pred)
        
        # Calculate multiple metrics
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        r2 = r2_score(y_test_orig, y_pred_orig)
        
        # Calculate correlation for each EMG feature
        correlations = []
        for i in range(y_test_orig.shape[1]):
            if np.std(y_test_orig[:, i]) > 0 and np.std(y_pred_orig[:, i]) > 0:
                corr, _ = pearsonr(y_test_orig[:, i], y_pred_orig[:, i])
                correlations.append(corr)
            else:
                correlations.append(0.0)
        
        avg_correlation = np.mean(correlations)
        
        model_results[name] = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'cv_r2_mean': np.mean(cv_scores),
            'cv_r2_std': np.std(cv_scores),
            'correlation': avg_correlation,
            'model': model
        }
        
        print(f"  Cross-validation R²: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
        print(f"  Test MSE: {mse:.6f}")
        print(f"  Test MAE: {mae:.6f}")
        print(f"  Test R²: {r2:.4f}")
        print(f"  Average Correlation: {avg_correlation:.4f}")
        
        # Update best model based on cross-validation R²
        if np.mean(cv_scores) > best_score:
            best_score = np.mean(cv_scores)
            best_model = model
            best_name = name
    
    print(f"\n=== BEST MODEL: {best_name} ===")
    print(f"Cross-validation R²: {model_results[best_name]['cv_r2_mean']:.4f} (±{model_results[best_name]['cv_r2_std']:.4f})")
    print(f"Test R²: {model_results[best_name]['r2']:.4f}")
    
    # Feature importance for tree-based models
    if hasattr(best_model, 'feature_importances_'):
        feature_names = list(ecg_df.columns)
        importance = best_model.feature_importances_
        feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 Most Important ECG Features ({best_name}):")
        for i, (name, imp) in enumerate(feature_importance[:10]):
            print(f"  {i+1:2d}. {name}: {imp:.4f}")
    
    # Detailed analysis for each EMG feature
    print(f"\n=== DETAILED EMG FEATURE ANALYSIS ===")
    emg_feature_names = list(emg_df.columns)
    y_test_orig = scaler_y.inverse_transform(y_test)
    y_pred_orig = scaler_y.inverse_transform(best_model.predict(X_test))
    
    for i, emg_feat in enumerate(emg_feature_names):
        if np.std(y_test_orig[:, i]) > 0:
            corr, p_value = pearsonr(y_test_orig[:, i], y_pred_orig[:, i])
            mse_feat = mean_squared_error(y_test_orig[:, i], y_pred_orig[:, i])
            r2_feat = r2_score(y_test_orig[:, i], y_pred_orig[:, i])
            print(f"  {emg_feat}:")
            print(f"    R²: {r2_feat:.4f}, Correlation: {corr:.4f}, MSE: {mse_feat:.6f}")
    
    # Save the best model
    joblib.dump({
        'model': best_model,
        'model_name': best_name,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_names': list(ecg_df.columns),
        'emg_feature_names': list(emg_df.columns),
        'model_results': model_results,
        'config': Config()
    }, 'best_emg_prediction_model.pkl')
    
    print(f"\nBest model ({best_name}) saved to 'best_emg_prediction_model.pkl'")
    
    # Create summary report
    print(f"\n=== SUMMARY REPORT ===")
    print(f"Dataset: {len(X)} valid records processed")
    print(f"ECG features: {X.shape[1]} features")
    print(f"EMG features: {y.shape[1]} features predicted")
    print(f"Best model: {best_name}")
    print(f"Best cross-validation R²: {best_score:.4f}")
    
    # Data quality insights
    print(f"\n=== DATA QUALITY INSIGHTS ===")
    print(f"ECG feature correlations with EMG:")
    for i, ecg_feat in enumerate(list(ecg_df.columns)[:5]):  # Top 5 ECG features
        avg_corr = np.mean([abs(pearsonr(X_clean[:, i], y_clean[:, j])[0]) 
                           for j in range(y_clean.shape[1]) 
                           if np.std(X_clean[:, i]) > 0 and np.std(y_clean[:, j]) > 0])
        print(f"  {ecg_feat}: {avg_corr:.4f}")

if __name__ == '__main__':
    main()