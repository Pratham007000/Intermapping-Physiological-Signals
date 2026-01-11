#!/usr/bin/env python3
"""
Arduino Nano PPG Plotter
========================
Plot PPG data from arduino_nano_csv_playback.ino output

Features:
- Plot from sample data (paste data directly into script)
- Real-time plotting from serial port
- Save plots as images
- Multiple visualization modes

Usage:
python plot_arduino_ppg.py [--mode sample|serial] [--port COM3] [--save filename.png]
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys
from datetime import datetime
import serial
import threading
import queue
import time

# Sample data from your Arduino output
SAMPLE_DATA = """10182,2253,341.0000,0.4665,60.0
10189,2254,365.0000,0.8279,60.0
10192,2255,365.0000,0.7364,60.0
10195,2256,355.0000,0.7057,60.0
10198,2257,375.0000,0.9055,60.0
10201,2258,329.0000,0.5595,60.0
10203,2259,379.0000,1.1194,60.0
10209,2260,309.0000,0.4788,60.0
10215,2261,390.0000,1.3436,60.0
10223,2262,294.0000,0.3904,60.0
10229,2263,402.0000,1.5000,60.0
10232,2264,282.0000,0.3099,60.0
10235,2265,406.0000,1.5000,60.0
10238,2266,272.0000,0.2744,60.0
10244,2267,369.0000,1.5000,60.0"""

def parse_arduino_data(data_string):
    """Parse Arduino CSV data into pandas DataFrame"""
    lines = data_string.strip().split('\n')
    
    data = []
    for line in lines:
        if line.strip() and ',' in line:
            parts = line.split(',')
            if len(parts) >= 5:
                timestamp = int(parts[0])
                sample_index = int(parts[1])
                ecg = float(parts[2])
                ppg = float(parts[3])
                heart_rate = float(parts[4])
                
                data.append({
                    'timestamp': timestamp,
                    'sample_index': sample_index,
                    'ecg': ecg,
                    'ppg': ppg,
                    'heart_rate': heart_rate
                })
    
    return pd.DataFrame(data)

def plot_ppg_data(df, title="PPG Signal from Arduino Nano", save_path=None):
    """Create comprehensive PPG visualization"""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Convert timestamp to relative seconds
    df['time_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0
    
    # Plot 1: PPG Signal
    axes[0].plot(df['time_seconds'], df['ppg'], 'b-', linewidth=2, label='PPG Signal')
    axes[0].set_title('PPG (Photoplethysmogram) Signal')
    axes[0].set_ylabel('PPG Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Add PPG statistics
    ppg_mean = df['ppg'].mean()
    ppg_std = df['ppg'].std()
    ppg_min = df['ppg'].min()
    ppg_max = df['ppg'].max()
    
    axes[0].axhline(y=ppg_mean, color='red', linestyle='--', alpha=0.7, label=f'Mean: {ppg_mean:.3f}')
    axes[0].fill_between(df['time_seconds'], ppg_mean - ppg_std, ppg_mean + ppg_std, 
                        alpha=0.2, color='red', label=f'±1σ: {ppg_std:.3f}')
    axes[0].legend()
    
    # Plot 2: ECG Signal for comparison
    axes[1].plot(df['time_seconds'], df['ecg'], 'g-', linewidth=1, label='ECG Signal')
    axes[1].set_title('ECG (Electrocardiogram) Signal')
    axes[1].set_ylabel('ECG Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Heart Rate
    axes[2].plot(df['time_seconds'], df['heart_rate'], 'r-', linewidth=2, marker='o', markersize=4)
    axes[2].set_title('Heart Rate Estimate')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Heart Rate (BPM)')
    axes[2].grid(True, alpha=0.3)
    
    # Add heart rate statistics
    hr_mean = df['heart_rate'].mean()
    axes[2].axhline(y=hr_mean, color='red', linestyle='--', alpha=0.7, label=f'Average: {hr_mean:.1f} BPM')
    axes[2].legend()
    
    plt.tight_layout()
    
    # Print statistics
    print("\n=== PPG Signal Analysis ===")
    print(f"Duration: {df['time_seconds'].iloc[-1]:.2f} seconds")
    print(f"Samples: {len(df)}")
    print(f"Sample rate: ~{len(df) / df['time_seconds'].iloc[-1]:.1f} Hz")
    print(f"\nPPG Statistics:")
    print(f"  Mean: {ppg_mean:.4f}")
    print(f"  Std:  {ppg_std:.4f}")
    print(f"  Min:  {ppg_min:.4f}")
    print(f"  Max:  {ppg_max:.4f}")
    print(f"  Range: {ppg_max - ppg_min:.4f}")
    print(f"\nHeart Rate:")
    print(f"  Average: {hr_mean:.1f} BPM")
    print(f"  Range: {df['heart_rate'].min():.1f} - {df['heart_rate'].max():.1f} BPM")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    return fig, axes

class RealTimeArduinoPlotter:
    """Real-time plotting of Arduino serial data"""
    
    def __init__(self, port, baudrate=9600, max_points=1000):
        self.port = port
        self.baudrate = baudrate
        self.max_points = max_points
        
        self.data_queue = queue.Queue()
        self.serial_thread = None
        self.running = False
        
        # Data storage
        self.timestamps = []
        self.ppg_data = []
        self.ecg_data = []
        self.heart_rates = []
        
    def start_serial_reader(self):
        """Start reading from serial port in separate thread"""
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Connected to {self.port} at {self.baudrate} baud")
            
            while self.running:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if line and ',' in line:
                        self.data_queue.put(line)
                except Exception as e:
                    print(f"Serial read error: {e}")
                    break
                    
            ser.close()
            
        except Exception as e:
            print(f"Could not open serial port {self.port}: {e}")
            self.running = False
    
    def process_data(self):
        """Process data from queue and update plots"""
        while not self.data_queue.empty():
            try:
                line = self.data_queue.get_nowait()
                parts = line.split(',')
                
                if len(parts) >= 5:
                    timestamp = int(parts[0])
                    ecg = float(parts[2])
                    ppg = float(parts[3])
                    heart_rate = float(parts[4])
                    
                    self.timestamps.append(timestamp)
                    self.ecg_data.append(ecg)
                    self.ppg_data.append(ppg)
                    self.heart_rates.append(heart_rate)
                    
                    # Keep only recent data
                    if len(self.timestamps) > self.max_points:
                        self.timestamps.pop(0)
                        self.ecg_data.pop(0)
                        self.ppg_data.pop(0)
                        self.heart_rates.pop(0)
                        
            except (ValueError, IndexError) as e:
                print(f"Data parsing error: {e}")
                continue
    
    def start_realtime_plot(self):
        """Start real-time plotting"""
        self.running = True
        
        # Start serial reader thread
        self.serial_thread = threading.Thread(target=self.start_serial_reader)
        self.serial_thread.daemon = True
        self.serial_thread.start()
        
        # Setup plot
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle('Real-time Arduino PPG Monitor', fontsize=16)
        
        line1, = ax1.plot([], [], 'b-', linewidth=2)
        ax1.set_title('PPG Signal')
        ax1.set_ylabel('PPG Amplitude')
        ax1.grid(True, alpha=0.3)
        
        line2, = ax2.plot([], [], 'g-', linewidth=1)
        ax2.set_title('ECG Signal')
        ax2.set_ylabel('ECG Amplitude')
        ax2.grid(True, alpha=0.3)
        
        line3, = ax3.plot([], [], 'r-', linewidth=2, marker='o', markersize=3)
        ax3.set_title('Heart Rate')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Heart Rate (BPM)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        try:
            while self.running:
                self.process_data()
                
                if len(self.timestamps) > 1:
                    # Convert to relative time
                    times = [(t - self.timestamps[0]) / 1000.0 for t in self.timestamps]
                    
                    # Update PPG plot
                    line1.set_data(times, self.ppg_data)
                    ax1.relim()
                    ax1.autoscale_view()
                    
                    # Update ECG plot
                    line2.set_data(times, self.ecg_data)
                    ax2.relim()
                    ax2.autoscale_view()
                    
                    # Update heart rate plot
                    line3.set_data(times, self.heart_rates)
                    ax3.relim()
                    ax3.autoscale_view()
                    
                    plt.pause(0.01)
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nStopping real-time plot...")
            
        finally:
            self.running = False
            plt.ioff()
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot PPG data from Arduino Nano')
    parser.add_argument('--mode', choices=['sample', 'serial'], default='sample',
                       help='Data source mode: sample data or serial port')
    parser.add_argument('--port', default='COM3' if sys.platform.startswith('win') else '/dev/ttyACM0',
                       help='Serial port for Arduino (default: COM3 on Windows, /dev/ttyACM0 on Linux)')
    parser.add_argument('--baudrate', type=int, default=9600,
                       help='Serial baudrate (default: 9600)')
    parser.add_argument('--save', help='Save plot to file (e.g., ppg_plot.png)')
    
    args = parser.parse_args()
    
    if args.mode == 'sample':
        # Use sample data
        print("Using sample Arduino data...")
        df = parse_arduino_data(SAMPLE_DATA)
        
        if df.empty:
            print("No valid data found in sample data!")
            return
            
        plot_ppg_data(df, "PPG Signal from Arduino Nano (Sample Data)", args.save)
        
    elif args.mode == 'serial':
        # Real-time serial plotting
        print(f"Starting real-time plotting from {args.port}...")
        print("Make sure your Arduino is connected and sending data.")
        print("Press Ctrl+C to stop.")
        
        plotter = RealTimeArduinoPlotter(args.port, args.baudrate)
        plotter.start_realtime_plot()

if __name__ == "__main__":
    main()
