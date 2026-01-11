#!/usr/bin/env python3
"""
Capture Arduino Serial Output to CSV in Downloads Folder
======================================================
This script reads data from the Arduino serial port, saves it to a CSV file
in the user's Downloads folder, and creates a compatible CSV for plot_arduino_final.py.
"""

import serial
import pandas as pd
import os
import sys
import time
import re
import subprocess
from pathlib import Path

def get_downloads_folder():
    """Get the path to the user's Downloads folder."""
    home = Path.home()
    downloads = home / "Downloads"
    if not downloads.exists():
        raise FileNotFoundError(f"Downloads folder not found at {downloads}")
    return downloads

def capture_serial_to_csv(port, baudrate, duration_seconds=10):
    """
    Capture serial data from Arduino and save to CSV in Downloads folder.
    
    Parameters:
    port (str): Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux/Mac)
    baudrate (int): Baud rate (e.g., 9600)
    duration_seconds (int): Duration to capture data (seconds)
    """
    try:
        # Get Downloads folder
        downloads_dir = get_downloads_folder()
        
        # Generate output filenames with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_csv = downloads_dir / f"arduino_data_{timestamp}.csv"
        plot_csv = downloads_dir / f"arduino_data_plot_{timestamp}.csv"
        
        # Initialize lists for data
        timestamps = []
        sample_indices = []
        ecg_values = []
        ppg_values = []
        heart_rates = []
        
        # Regular expression for format: timestamp,sample_index,ecg,ppg,heart_rate
        pattern = r'(\d+),(\d+),(\d+\.\d+),(\d+\.\d+),(\d+\.\d+)'
        
        # Open serial port
        with serial.Serial(port, baudrate, timeout=1) as ser:
            print(f"✅ Connected to {port} at {baudrate} baud")
            print(f"📝 Capturing data for {duration_seconds} seconds...")
            
            start_time = time.time()
            line_count = 0
            while time.time() - start_time < duration_seconds:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    match = re.match(pattern, line)
                    if match:
                        timestamps.append(int(match.group(1)))
                        sample_indices.append(int(match.group(2)))
                        ecg_values.append(float(match.group(3)))
                        ppg_values.append(float(match.group(4)))
                        heart_rates.append(float(match.group(5)))
                        line_count += 1
                    else:
                        print(f"⚠️ Skipped invalid line: {line}")
        
        if not timestamps:
            raise ValueError("No valid data captured")
        
        # Create DataFrame with all columns
        df = pd.DataFrame({
            'timestamp': timestamps,
            'sample_index': sample_indices,
            'ecg': ecg_values,
            'ppg': ppg_values,
            'heart_rate': heart_rates
        })
        
        # Save full CSV to Downloads
        df.to_csv(output_csv, index=False)
        print(f"✅ Saved full data to {output_csv} ({len(df)} samples)")
        
        # Create plot-compatible CSV (timestamp,ecg,ppg only)
        plot_df = df[['timestamp', 'ecg', 'ppg']].copy()
        plot_df.to_csv(plot_csv, index=False)
        print(f"✅ Saved plot-compatible data to {plot_csv} ({len(plot_df)} samples)")
        
        return output_csv, plot_csv
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

def run_plot_script(plot_csv):
    """
    Run the plot_arduino_final.py script with the plot-compatible CSV file.
    """
    try:
        plot_script = "plot_arduino_final.py"
        if not os.path.exists(plot_script):
            raise FileNotFoundError(f"Plot script {plot_script} not found in current directory")
        
        # Run the plot script
        result = subprocess.run(['python3', plot_script, str(plot_csv)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Plotting completed successfully")
            print(result.stdout)
        else:
            print(f"❌ Plotting failed")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running plot script: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 capture_arduino_to_csv.py <port> <baudrate> [duration_seconds]")
        print("Example: python3 capture_arduino_to_csv.py COM3 9600 10")
        sys.exit(1)
    
    port = sys.argv[1]
    baudrate = int(sys.argv[2])
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    try:
        # Capture data and save to CSV
        output_csv, plot_csv = capture_serial_to_csv(port, baudrate, duration)
        
        # Run the plotting script
        run_plot_script(plot_csv)
        
        print(f"\n🎉 Processing complete! CSV files saved to Downloads folder:")
        print(f"   - Full data: {output_csv}")
        print(f"   - Plot data: {plot_csv}")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)