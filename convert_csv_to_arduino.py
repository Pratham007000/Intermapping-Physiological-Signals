#!/usr/bin/env python3
"""
Convert CSV ECG Data to Arduino Header File
==========================================

This script converts the CSV ECG data file into a C header file that can be
included in Arduino sketches. The ECG data is stored in PROGMEM (flash memory)
to save precious SRAM on the Arduino Nano 33 BLE Sense.

Usage:
    python convert_csv_to_arduino.py

Output:
    ecg_data_array.h - Header file for Arduino
"""

import pandas as pd
import numpy as np
import os

def convert_csv_to_arduino_header(csv_file, header_file, max_samples=None, downsample_factor=2):
    """
    Convert CSV ECG data to Arduino header file.
    
    Args:
        csv_file: Path to input CSV file
        header_file: Path to output header file  
        max_samples: Maximum number of samples to include (None for all)
        downsample_factor: Factor to downsample data (e.g., 2 = take every 2nd sample)
    """
    
    print(f"Converting {csv_file} to Arduino header...")
    
    try:
        # Read CSV file
        data = pd.read_csv(csv_file)
        print(f"Loaded CSV with {len(data)} rows and columns: {list(data.columns)}")
        
        # Extract ECG data (assuming column is named "ECG Amplitude" or similar)
        ecg_columns = [col for col in data.columns if 'ECG' in col.upper()]
        if not ecg_columns:
            # Try other common column names
            ecg_columns = [col for col in data.columns if any(name in col.upper() for name in ['ECG', 'AMPLITUDE', 'SIGNAL'])]
        
        if not ecg_columns:
            print("Available columns:", list(data.columns))
            ecg_column = input("Enter the ECG column name: ")
        else:
            ecg_column = ecg_columns[0]
            print(f"Using ECG column: {ecg_column}")
        
        # Extract ECG values
        ecg_values = data[ecg_column].values
        
        # Apply downsampling
        if downsample_factor > 1:
            ecg_values = ecg_values[::downsample_factor]
            print(f"Downsampled by factor {downsample_factor}: {len(ecg_values)} samples")
        
        # Limit number of samples if specified
        if max_samples and len(ecg_values) > max_samples:
            ecg_values = ecg_values[:max_samples]
            print(f"Limited to {max_samples} samples")
        
        # Convert to float32 and normalize
        ecg_values = ecg_values.astype(np.float32)
        
        # Basic statistics
        print(f"ECG data statistics:")
        print(f"  Samples: {len(ecg_values)}")
        print(f"  Range: {ecg_values.min():.4f} to {ecg_values.max():.4f}")
        print(f"  Mean: {ecg_values.mean():.4f}")
        print(f"  Std: {ecg_values.std():.4f}")
        
        # Calculate memory usage
        memory_usage = len(ecg_values) * 4  # 4 bytes per float32
        print(f"  Memory usage: {memory_usage:,} bytes ({memory_usage/1024:.1f} KB)")
        
        # Check if it fits in Arduino memory
        if memory_usage > 200000:  # Conservative limit for Nano 33 BLE
            print("⚠️  WARNING: Data may be too large for Arduino memory!")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        # Generate header file
        with open(header_file, 'w') as f:
            f.write("/*\n")
            f.write(" * Arduino ECG Data Array\n")
            f.write(" * Generated from CSV file\n")
            f.write(" * \n")
            f.write(f" * Original file: {os.path.basename(csv_file)}\n")
            f.write(f" * Samples: {len(ecg_values):,}\n")
            f.write(f" * Memory usage: {memory_usage:,} bytes\n")
            f.write(f" * Downsample factor: {downsample_factor}\n")
            f.write(" * Storage: PROGMEM (Flash memory)\n")
            f.write(" */\n\n")
            
            f.write("#ifndef ECG_DATA_ARRAY_H\n")
            f.write("#define ECG_DATA_ARRAY_H\n\n")
            
            f.write("#include <Arduino.h>\n")
            f.write("#include <avr/pgmspace.h>\n\n")
            
            f.write(f"#define ECG_DATA_SIZE {len(ecg_values)}\n\n")
            
            f.write("// ECG data stored in PROGMEM (flash memory)\n")
            f.write("const PROGMEM float ecg_data_array[ECG_DATA_SIZE] = {\n")
            
            # Write data in rows of 8 values for readability
            for i in range(0, len(ecg_values), 8):
                row = ecg_values[i:i+8]
                values_str = ", ".join(f"{val:.6f}f" for val in row)
                f.write(f"    {values_str}")
                
                if i + 8 < len(ecg_values):
                    f.write(",\n")
                else:
                    f.write("\n")
            
            f.write("};\n\n")
            f.write("#endif // ECG_DATA_ARRAY_H\n")
        
        print(f"✅ Arduino header file created: {header_file}")
        print(f"📁 File size: {os.path.getsize(header_file):,} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "ecg_data_20250701_172937.csv")
    header_file = os.path.join(script_dir, "arduino_nano_csv_playback", "ecg_data_array.h")
    
    # Parameters for Arduino optimization
    MAX_SAMPLES = 10000  # Limit to ~40KB of data
    DOWNSAMPLE_FACTOR = 2  # Take every 2nd sample to reduce size
    
    print("🔄 Converting CSV ECG data to Arduino header file")
    print("=" * 50)
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        print("Available files:")
        for f in os.listdir(script_dir):
            if f.endswith('.csv'):
                print(f"  - {f}")
        
        csv_file = input("Enter the CSV filename: ")
        if not os.path.exists(csv_file):
            csv_file = os.path.join(script_dir, csv_file)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(header_file), exist_ok=True)
    
    # Convert the file
    success = convert_csv_to_arduino_header(
        csv_file=csv_file,
        header_file=header_file,
        max_samples=MAX_SAMPLES,
        downsample_factor=DOWNSAMPLE_FACTOR
    )
    
    if success:
        print("\n🎉 Conversion completed successfully!")
        print("\nNext steps:")
        print("1. Copy the ecg_data_array.h file to your Arduino sketch folder")
        print("2. Use the arduino_nano_csv_playback.ino sketch")  
        print("3. Upload to Arduino Nano 33 BLE Sense")
        print("4. Monitor serial output to see ECG data playback")
        
        # Create the full Arduino folder structure
        arduino_folder = os.path.dirname(header_file)
        
        # Copy required files
        required_files = [
            "arduino_nano_tinyml_model.h",
            "arduino_nano_tinyml_model.cpp", 
            "arduino_nano_model_weights.cpp"
        ]
        
        for file in required_files:
            src = os.path.join(script_dir, "arduino_nano_main", file)
            dst = os.path.join(arduino_folder, file)
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, dst)
                print(f"✅ Copied {file}")
        
        print(f"\n📂 Complete Arduino sketch ready in: {arduino_folder}")
        
    else:
        print("\n❌ Conversion failed!")

if __name__ == "__main__":
    main()
