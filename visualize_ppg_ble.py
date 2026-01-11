#!/usr/bin/env python3
"""
PPG Data Visualization via Bluetooth Low Energy
==============================================

This script connects to Arduino Nano 33 BLE Sense via Bluetooth
and creates live graphs of ECG input and PPG predictions.

Requirements:
    pip install matplotlib numpy bleak asyncio

Usage:
    python3 visualize_ppg_ble.py
"""

import asyncio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import time
from bleak import BleakClient, BleakScanner
import struct

# Arduino BLE Service UUIDs (from Arduino code)
PPG_SERVICE_UUID = "12345678-1234-1234-1234-123456789abc"
ECG_CHAR_UUID = "12345678-1234-1234-1234-123456789abd" 
PPG_CHAR_UUID = "12345678-1234-1234-1234-123456789abe"
HR_CHAR_UUID = "12345678-1234-1234-1234-123456789abf"

# Configuration
WINDOW_SIZE = 1000  # Number of samples to display
UPDATE_INTERVAL = 50  # milliseconds

# Data storage
timestamps = deque(maxlen=WINDOW_SIZE)
ecg_values = deque(maxlen=WINDOW_SIZE)
ppg_values = deque(maxlen=WINDOW_SIZE) 
heart_rates = deque(maxlen=WINDOW_SIZE)

# Global variables
client = None
start_time = time.time()
latest_ecg = 0
latest_ppg = 0
latest_hr = 60

async def find_arduino():
    """Scan for Arduino BLE device"""
    print("Scanning for Arduino TinyML device...")
    
    devices = await BleakScanner.discover(timeout=10.0)
    
    for device in devices:
        if device.name and "TinyML" in device.name:
            print(f"Found Arduino: {device.name} ({device.address})")
            return device.address
    
    print("Arduino not found. Make sure it's powered on and advertising.")
    print("Looking for devices with 'TinyML' in the name...")
    print("Available devices:")
    for device in devices:
        if device.name:
            print(f"  {device.name} ({device.address})")
    
    return None

def ecg_notification_handler(sender, data):
    """Handle ECG data notifications"""
    global latest_ecg, start_time
    try:
        # Arduino sends int32_t scaled by 1000
        ecg_int = struct.unpack('<i', data)[0]  # little-endian int32
        latest_ecg = ecg_int / 1000.0  # Convert back to float
        
        # Add timestamp
        current_time = time.time() - start_time
        
    except Exception as e:
        print(f"Error parsing ECG data: {e}")

def ppg_notification_handler(sender, data):
    """Handle PPG data notifications"""
    global latest_ppg, start_time
    try:
        # Arduino sends int32_t scaled by 1000
        ppg_int = struct.unpack('<i', data)[0]  # little-endian int32
        latest_ppg = ppg_int / 1000.0  # Convert back to float
        
        # Add all current values to deques (sync point)
        current_time = time.time() - start_time
        timestamps.append(current_time)
        ecg_values.append(latest_ecg)
        ppg_values.append(latest_ppg)
        heart_rates.append(latest_hr)
        
    except Exception as e:
        print(f"Error parsing PPG data: {e}")

def hr_notification_handler(sender, data):
    """Handle heart rate data notifications"""
    global latest_hr
    try:
        # Arduino sends int32_t heart rate
        hr_int = struct.unpack('<i', data)[0]  # little-endian int32
        latest_hr = float(hr_int)
        
    except Exception as e:
        print(f"Error parsing heart rate data: {e}")

async def connect_to_arduino():
    """Connect to Arduino via BLE"""
    global client
    
    address = await find_arduino()
    if not address:
        return False
    
    try:
        client = BleakClient(address)
        await client.connect()
        print(f"Connected to Arduino at {address}")
        
        # Subscribe to notifications
        await client.start_notify(ECG_CHAR_UUID, ecg_notification_handler)
        await client.start_notify(PPG_CHAR_UUID, ppg_notification_handler)
        await client.start_notify(HR_CHAR_UUID, hr_notification_handler)
        
        print("Subscribed to BLE notifications")
        return True
        
    except Exception as e:
        print(f"Failed to connect: {e}")
        return False

def update_plot(frame):
    """Update the real-time plot"""
    global ax1, ax2
    
    if len(timestamps) > 10:  # Wait for some data
        times = list(timestamps)
        
        # Clear and update ECG plot
        ax1.clear()
        ax1.plot(times, list(ecg_values), 'b-', label='ECG Input', linewidth=1)
        ax1.set_ylabel('ECG Amplitude', color='b')
        ax1.set_title('Real-time ECG Input vs PPG Prediction (BLE)')
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
    
    return ax1, ax2

def save_data_to_file():
    """Save collected data to CSV file"""
    if len(timestamps) > 0:
        filename = f"ppg_ble_data_{int(time.time())}.csv"
        with open(filename, 'w') as f:
            f.write("timestamp,ecg,ppg,heart_rate\n")
            for i in range(len(timestamps)):
                f.write(f"{timestamps[i]:.3f},{ecg_values[i]:.4f},"
                       f"{ppg_values[i]:.4f},{heart_rates[i]:.1f}\n")
        print(f"Data saved to {filename}")

async def main():
    """Main async function"""
    global ax1, ax2, fig, client
    
    print("PPG BLE Data Visualization Tool")
    print("==============================")
    
    # Connect to Arduino
    if not await connect_to_arduino():
        return
    
    # Setup matplotlib
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Arduino TinyML: ECG → PPG Real-time Inference (BLE)', fontsize=16)
    
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
        print("\nBLE Visualization started!")
        print("- Blue line: ECG input from CSV data")
        print("- Red line: PPG prediction from TinyML model")
        print("- Data received via Bluetooth Low Energy")
        print("- Close window or Ctrl+C to stop")
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if client and client.is_connected:
            await client.disconnect()
        save_data_to_file()
        print("BLE connection closed!")

def run_ble_visualization():
    """Run the async BLE visualization"""
    try:
        asyncio.run(main())
    except KeyboardboardInterrupt:
        print("Interrupted!")

if __name__ == "__main__":
    run_ble_visualization()
