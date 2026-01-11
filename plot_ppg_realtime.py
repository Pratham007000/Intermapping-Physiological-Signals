#!/usr/bin/env python3
"""
Real-time PPG Plotter for Arduino TinyML ECG-to-PPG System
==========================================================

This script reads data from the Arduino Nano 33 BLE Sense running the TinyML
ECG-to-PPG conversion model and plots the signals in real-time.

Data format expected: timestamp,sample_index,ecg,ppg,heart_rate

Usage:
    python plot_ppg_realtime.py [--port /dev/cu.usbmodem...] [--baud 115200]
"""

import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import argparse
import time
import sys
import re

class PPGPlotter:
    def __init__(self, port, baud_rate=115200, max_points=1000):
        self.port = port
        self.baud_rate = baud_rate
        self.max_points = max_points
        
        # Data storage
        self.timestamps = deque(maxlen=max_points)
        self.ecg_data = deque(maxlen=max_points)
        self.ppg_data = deque(maxlen=max_points)
        self.heart_rates = deque(maxlen=max_points)
        
        # Serial connection
        self.serial_conn = None
        self.connect_serial()
        
        # Setup plotting
        self.setup_plot()
        
        # Statistics
        self.total_samples = 0
        self.start_time = time.time()
        self.last_heart_rate = 60.0
        
    def connect_serial(self):
        """Connect to Arduino serial port"""
        try:
            print(f"Connecting to Arduino on {self.port} at {self.baud_rate} baud...")
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # Give Arduino time to reset
            print("✅ Connected to Arduino!")
            
            # Clear any initial data
            self.serial_conn.flushInput()
            
        except serial.SerialException as e:
            print(f"❌ Failed to connect to {self.port}: {e}")
            print("\nAvailable ports:")
            self.list_ports()
            sys.exit(1)
    
    def list_ports(self):
        """List available serial ports"""
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(f"  {port.device} - {port.description}")
    
    def setup_plot(self):
        """Setup matplotlib real-time plot"""
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # ECG subplot
        self.ax1.set_title('ECG Signal (Input)', fontsize=14, color='white')
        self.ax1.set_ylabel('ECG Amplitude', color='cyan')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_facecolor('black')
        self.line_ecg, = self.ax1.plot([], [], 'cyan', linewidth=1.5, label='ECG')
        self.ax1.legend(loc='upper right')
        
        # PPG subplot  
        self.ax2.set_title('PPG Signal (TinyML LSTM Output)', fontsize=14, color='white')
        self.ax2.set_xlabel('Time (seconds)', color='white')
        self.ax2.set_ylabel('PPG Amplitude', color='red')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_facecolor('black')
        self.line_ppg, = self.ax2.plot([], [], 'red', linewidth=2, label='PPG')
        self.ax2.legend(loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        
        # Add text for heart rate display
        self.hr_text = self.ax2.text(0.02, 0.95, '', transform=self.ax2.transAxes, 
                                     fontsize=12, color='yellow', 
                                     bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
    
    def parse_line(self, line):
        """Parse Arduino CSV output line"""
        try:
            # Look for CSV format: timestamp,sample_index,ecg,ppg,heart_rate
            pattern = r'(\d+),(\d+),(\d+\.\d+),(\d+\.\d+),(\d+\.\d+)'
            match = re.search(pattern, line)
            
            if match:
                timestamp = int(match.group(1))
                sample_index = int(match.group(2))
                ecg = float(match.group(3))
                ppg = float(match.group(4))
                heart_rate = float(match.group(5))
                
                return timestamp, sample_index, ecg, ppg, heart_rate
                
        except (ValueError, IndexError) as e:
            pass  # Skip malformed lines
            
        return None
    
    def read_data(self):
        """Read and parse data from Arduino"""
        if not self.serial_conn or not self.serial_conn.in_waiting:
            return False
            
        try:
            line = self.serial_conn.readline().decode('utf-8').strip()
            
            # Skip debug messages and other non-data lines
            if not line or 'DEBUG' in line or 'WARNING' in line or 'SUCCESS' in line:
                return False
                
            parsed = self.parse_line(line)
            if parsed:
                timestamp, sample_index, ecg, ppg, heart_rate = parsed
                
                # Convert timestamp to relative seconds
                if not self.timestamps:
                    self.start_timestamp = timestamp
                
                rel_time = (timestamp - self.start_timestamp) / 1000.0
                
                # Store data
                self.timestamps.append(rel_time)
                self.ecg_data.append(ecg)
                self.ppg_data.append(ppg)
                self.heart_rates.append(heart_rate)
                
                self.total_samples += 1
                self.last_heart_rate = heart_rate
                
                return True
                
        except (UnicodeDecodeError, serial.SerialException):
            pass
            
        return False
    
    def animate(self, frame):
        """Animation function for real-time plotting"""
        # Read new data
        data_updated = False
        for _ in range(10):  # Read up to 10 samples per frame
            if self.read_data():
                data_updated = True
            else:
                break
        
        if not data_updated or len(self.timestamps) < 2:
            return self.line_ecg, self.line_ppg, self.hr_text
        
        # Convert to numpy arrays for plotting
        times = np.array(self.timestamps)
        ecg = np.array(self.ecg_data)
        ppg = np.array(self.ppg_data)
        
        # Update ECG plot
        self.line_ecg.set_data(times, ecg)
        self.ax1.set_xlim(times[-1] - 10, times[-1] + 1)  # Show last 10 seconds
        if len(ecg) > 0:
            ecg_min, ecg_max = np.min(ecg[-360*10:]), np.max(ecg[-360*10:])  # Last 10 sec
            ecg_range = ecg_max - ecg_min
            self.ax1.set_ylim(ecg_min - 0.1*ecg_range, ecg_max + 0.1*ecg_range)
        
        # Update PPG plot
        self.line_ppg.set_data(times, ppg)
        self.ax2.set_xlim(times[-1] - 10, times[-1] + 1)  # Show last 10 seconds
        if len(ppg) > 0:
            ppg_min, ppg_max = np.min(ppg[-360*10:]), np.max(ppg[-360*10:])  # Last 10 sec
            ppg_range = ppg_max - ppg_min
            if ppg_range > 0:
                self.ax2.set_ylim(ppg_min - 0.1*ppg_range, ppg_max + 0.1*ppg_range)
            else:
                self.ax2.set_ylim(0, 2)
        
        # Update heart rate display
        elapsed = time.time() - self.start_time
        sample_rate = self.total_samples / elapsed if elapsed > 0 else 0
        
        hr_text = f'Heart Rate: {self.last_heart_rate:.1f} BPM\n'
        hr_text += f'Samples: {self.total_samples}\n'
        hr_text += f'Sample Rate: {sample_rate:.1f} Hz\n'
        hr_text += f'Runtime: {elapsed:.1f}s'
        
        self.hr_text.set_text(hr_text)
        
        return self.line_ecg, self.line_ppg, self.hr_text
    
    def start_plotting(self):
        """Start real-time plotting"""
        print("🚀 Starting real-time PPG plotting...")
        print("📊 Waiting for data from Arduino...")
        
        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig, self.animate, interval=50, blit=False, cache_frame_data=False
        )
        
        # Show plot
        plt.show()
    
    def cleanup(self):
        """Clean up resources"""
        if self.serial_conn:
            self.serial_conn.close()
        print("🧹 Cleaned up resources")

def find_arduino_port():
    """Automatically find Arduino port"""
    import serial.tools.list_ports
    
    # Common Arduino identifiers
    arduino_identifiers = [
        'Arduino', 'CH340', 'CP2102', 'FT232', 'usbmodem', 'usbserial'
    ]
    
    ports = serial.tools.list_ports.comports()
    for port in ports:
        for identifier in arduino_identifiers:
            if identifier.lower() in port.description.lower() or identifier.lower() in port.device.lower():
                return port.device
                
    return None

def main():
    parser = argparse.ArgumentParser(description='Real-time PPG plotter for Arduino TinyML')
    parser.add_argument('--port', '-p', help='Serial port (e.g., /dev/cu.usbmodem...)')
    parser.add_argument('--baud', '-b', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('--max-points', '-m', type=int, default=3600, help='Maximum data points to store (default: 3600)')
    
    args = parser.parse_args()
    
    # Find Arduino port if not specified
    if not args.port:
        auto_port = find_arduino_port()
        if auto_port:
            print(f"🔍 Auto-detected Arduino on: {auto_port}")
            args.port = auto_port
        else:
            print("❌ Could not auto-detect Arduino port.")
            print("\nAvailable ports:")
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for port in ports:
                print(f"  {port.device} - {port.description}")
            print("\nPlease specify port with --port option")
            sys.exit(1)
    
    # Create plotter
    try:
        plotter = PPGPlotter(args.port, args.baud, args.max_points)
        plotter.start_plotting()
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'plotter' in locals():
            plotter.cleanup()

if __name__ == "__main__":
    main()
