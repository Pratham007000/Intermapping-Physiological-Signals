#!/usr/bin/env python3
"""
Real-time PPG Data Visualization from Arduino
===========================================

This script connects to Arduino Nano 33 BLE Sense via serial/USB
and creates live graphs of ECG input and PPG predictions.

Requirements:
    pip install matplotlib numpy pyserial

Usage:
    python3 visualize_ppg_data.py
"""

import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import time
import sys
import glob

# Configuration
SERIAL_BAUD_RATE = 9600
WINDOW_SIZE = 1000  # Number of samples to display
UPDATE_INTERVAL = 50  # milliseconds

# Data storage
timestamps = deque(maxlen=WINDOW_SIZE)
ecg_values = deque(maxlen=WINDOW_SIZE)
ppg_values = deque(maxlen=WINDOW_SIZE)
heart_rates = deque(maxlen=WINDOW_SIZE)

# Serial connection
ser = None

def find_arduino_port():
    """Find Arduino serial port automatically"""
    ports = glob.glob('/dev/cu.usbmodem*')
    if not ports:
        ports = glob.glob('/dev/cu.usbserial*')
    if not ports:
        ports = glob.glob('/dev/tty.usbmodem*')
    
    if ports:
        print(f"Found potential Arduino ports: {ports}")
        return ports[0]
    return None

def connect_to_arduino():
    """Establish serial connection to Arduino"""
    global ser
    
    port = find_arduino_port()
    if not port:
        print("No Arduino found. Please check connection.")
        print("Available ports:")
        ports = glob.glob('/dev/cu.*')
        for p in ports[:10]:  # Show first 10 ports
            print(f"  {p}")
        return False
    
    try:
        ser = serial.Serial(port, SERIAL_BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for Arduino to initialize
        print(f"Connected to Arduino on {port}")
        return True
    except Exception as e:
        print(f"Failed to connect: {e}")
        return False

def parse_arduino_data(line):
    """Parse CSV data from Arduino: timestamp,sample_index,ecg,ppg,heart_rate"""
    try:
        parts = line.strip().split(',')
        if len(parts) >= 5:
            timestamp = float(parts[0]) / 1000.0  # Convert ms to seconds
            ecg = float(parts[2])
            ppg = float(parts[3])
            heart_rate = float(parts[4])
            return timestamp, ecg, ppg, heart_rate
    except (ValueError, IndexError):
        pass
    return None

def update_plot(frame):
    """Update the real-time plot"""
    global ser
    
    if ser and ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8')
            data = parse_arduino_data(line)
            
            if data:
                timestamp, ecg, ppg, heart_rate = data
                
                # Add to data collections
                timestamps.append(timestamp)
                ecg_values.append(ecg)
                ppg_values.append(ppg)
                heart_rates.append(heart_rate)
                
                # Update plots
                if len(timestamps) > 1:
                    times = list(timestamps)
                    
                    # Clear and update ECG plot
                    ax1.clear()
                    ax1.plot(times, list(ecg_values), 'b-', label='ECG Input', linewidth=1)
                    ax1.set_ylabel('ECG Amplitude', color='b')
                    ax1.set_title('Real-time ECG Input vs PPG Prediction')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend(loc='upper left')
                    
                    # Clear and update PPG plot
                    ax2.clear()
                    ax2.plot(times, list(ppg_values), 'r-', label='PPG Prediction', linewidth=1)
                    ax2.set_ylabel('PPG Amplitude', color='r')
                    ax2.set_xlabel('Time (seconds)')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(loc='upper right')
                    
                    # Update heart rate display
                    if heart_rates:
                        current_hr = heart_rates[-1]
                        ax1.text(0.02, 0.95, f'Heart Rate: {current_hr:.1f} BPM', 
                                transform=ax1.transAxes, fontsize=12, 
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                    
                    # Set consistent x-axis
                    if times:
                        ax1.set_xlim(times[0], times[-1])
                        ax2.set_xlim(times[0], times[-1])
                        
        except Exception as e:
            print(f"Error reading data: {e}")
    
    return ax1, ax2

def save_data_to_file():
    """Save collected data to CSV file"""
    if len(timestamps) > 0:
        filename = f"ppg_data_{int(time.time())}.csv"
        with open(filename, 'w') as f:
            f.write("timestamp,ecg,ppg,heart_rate\n")
            for i in range(len(timestamps)):
                f.write(f"{timestamps[i]:.3f},{ecg_values[i]:.4f},"
                       f"{ppg_values[i]:.4f},{heart_rates[i]:.1f}\n")
        print(f"Data saved to {filename}")

def main():
    """Main function"""
    global ax1, ax2, fig
    
    print("PPG Data Visualization Tool")
    print("==========================")
    
    # Connect to Arduino
    if not connect_to_arduino():
        return
    
    # Setup matplotlib
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Arduino TinyML: ECG → PPG Real-time Inference', fontsize=16)
    
    # Setup initial plots
    ax1.set_ylabel('ECG Amplitude')
    ax1.set_title('ECG Input Signal')
    ax2.set_ylabel('PPG Amplitude') 
    ax2.set_xlabel('Time (seconds)')
    ax2.set_title('PPG Prediction Output')
    
    plt.tight_layout()
    
    # Start animation
    ani = animation.FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL, 
                                 blit=False, repeat=True)
    
    try:
        print("\nVisualization started!")
        print("- Blue line: ECG input from CSV data")
        print("- Red line: PPG prediction from TinyML model")
        print("- Close window or Ctrl+C to stop")
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if ser:
            ser.close()
        save_data_to_file()
        print("Done!")

if __name__ == "__main__":
    main()
