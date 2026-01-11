#!/usr/bin/env python3
"""
Real-time Arduino PPG Serial Plotter
====================================
This script connects to your Arduino and plots PPG data in real-time
from the serial monitor output.

Format expected: timestamp,sample_index,ecg,ppg,heart_rate

Usage:
python3 arduino_ppg_realtime.py [--port PORT] [--baudrate RATE] [--save]
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import serial
import argparse
import sys
import time
from datetime import datetime
from collections import deque
import threading
import queue

class ArduinoPPGPlotter:
    def __init__(self, port, baudrate=9600, window_size=1000):
        self.port = port
        self.baudrate = baudrate
        self.window_size = window_size
        
        # Data storage
        self.timestamps = deque(maxlen=window_size)
        self.ecg_data = deque(maxlen=window_size)
        self.ppg_data = deque(maxlen=window_size)
        self.hr_data = deque(maxlen=window_size)
        
        # Serial communication
        self.serial_conn = None
        self.data_queue = queue.Queue()
        self.running = False
        self.serial_thread = None
        
        # Statistics
        self.start_time = time.time()
        self.sample_count = 0
        self.last_update = time.time()
        
        # Setup plot
        self.setup_plot()
        
    def setup_plot(self):
        """Initialize the matplotlib plot"""
        plt.style.use('default')
        self.fig, self.axes = plt.subplots(3, 1, figsize=(14, 10))
        self.fig.suptitle('Real-time Arduino PPG Monitor', fontsize=16, fontweight='bold')
        
        # PPG subplot
        self.axes[0].set_title('PPG Signal (Photoplethysmogram)', fontsize=12, fontweight='bold')
        self.axes[0].set_ylabel('PPG Amplitude')
        self.axes[0].grid(True, alpha=0.3)
        self.ppg_line, = self.axes[0].plot([], [], 'r-', linewidth=1.5, label='PPG Signal')
        self.axes[0].legend()
        
        # ECG subplot  
        self.axes[1].set_title('ECG Signal (Electrocardiogram)', fontsize=12, fontweight='bold')
        self.axes[1].set_ylabel('ECG Amplitude')
        self.axes[1].grid(True, alpha=0.3)
        self.ecg_line, = self.axes[1].plot([], [], 'g-', linewidth=1.5, label='ECG Signal')
        self.axes[1].legend()
        
        # Heart Rate subplot
        self.axes[2].set_title('Heart Rate Estimation', fontsize=12, fontweight='bold')
        self.axes[2].set_xlabel('Time (seconds)')
        self.axes[2].set_ylabel('Heart Rate (BPM)')
        self.axes[2].grid(True, alpha=0.3)
        self.hr_line, = self.axes[2].plot([], [], 'b-', linewidth=2, marker='o', markersize=3, label='Heart Rate')
        self.axes[2].set_ylim(40, 120)
        self.axes[2].legend()
        
        plt.tight_layout()
        
    def connect_serial(self):
        """Connect to Arduino via serial port"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"✅ Connected to Arduino on {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to {self.port}: {e}")
            return False
    
    def serial_reader(self):
        """Read data from serial port in separate thread"""
        while self.running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if line and ',' in line:
                        self.data_queue.put(line)
            except Exception as e:
                print(f"Serial read error: {e}")
                break
            time.sleep(0.001)  # Small delay to prevent CPU overload
    
    def process_serial_data(self):
        """Process incoming serial data"""
        processed = 0
        while not self.data_queue.empty() and processed < 10:  # Limit processing to prevent lag
            try:
                line = self.data_queue.get_nowait()
                parts = line.split(',')
                
                if len(parts) >= 5:
                    timestamp = float(parts[0]) / 1000.0  # Convert ms to seconds
                    sample_index = int(parts[1])
                    ecg = float(parts[2])
                    ppg = float(parts[3])
                    heart_rate = float(parts[4])
                    
                    # Convert to relative time
                    rel_time = timestamp - (self.timestamps[0] if self.timestamps else timestamp)
                    
                    self.timestamps.append(rel_time)
                    self.ecg_data.append(ecg)
                    self.ppg_data.append(ppg)
                    self.hr_data.append(heart_rate)
                    
                    self.sample_count += 1
                    processed += 1
                    
            except (ValueError, IndexError, queue.Empty):
                break
        
        return processed > 0
    
    def update_plot(self, frame):
        """Update plot with new data"""
        if not self.running:
            return self.ppg_line, self.ecg_line, self.hr_line
        
        # Process new data
        has_new_data = self.process_serial_data()
        
        if len(self.timestamps) > 1 and has_new_data:
            times = list(self.timestamps)
            
            # Update PPG plot
            self.ppg_line.set_data(times, list(self.ppg_data))
            self.axes[0].relim()
            self.axes[0].autoscale_view()
            
            # Update ECG plot
            self.ecg_line.set_data(times, list(self.ecg_data))
            self.axes[1].relim()
            self.axes[1].autoscale_view()
            
            # Update Heart Rate plot
            self.hr_line.set_data(times, list(self.hr_data))
            self.axes[2].relim()
            self.axes[2].autoscale_view()
            
            # Update statistics display
            current_time = time.time()
            if current_time - self.last_update > 5.0:  # Update stats every 5 seconds
                duration = current_time - self.start_time
                rate = self.sample_count / duration if duration > 0 else 0
                
                # Update title with statistics
                stats = f"Samples: {self.sample_count:,} | Rate: {rate:.1f} Hz"
                if self.ppg_data:
                    stats += f" | PPG: {np.mean(list(self.ppg_data)):.2f}±{np.std(list(self.ppg_data)):.2f}"
                if self.hr_data:
                    stats += f" | HR: {np.mean(list(self.hr_data)):.1f} BPM"
                
                self.fig.suptitle(f'Real-time Arduino PPG Monitor - {stats}', fontsize=14, fontweight='bold')
                self.last_update = current_time
        
        return self.ppg_line, self.ecg_line, self.hr_line
    
    def start_plotting(self):
        """Start the real-time plotting"""
        if not self.connect_serial():
            return False
        
        self.running = True
        
        # Start serial reader thread
        self.serial_thread = threading.Thread(target=self.serial_reader, daemon=True)
        self.serial_thread.start()
        
        print("🔥 Starting real-time PPG plotting...")
        print("📊 Format expected: timestamp,sample_index,ecg,ppg,heart_rate")
        print("⏹️  Close the plot window or press Ctrl+C to stop")
        
        try:
            # Start animation
            ani = animation.FuncAnimation(
                self.fig, self.update_plot, interval=50, blit=False, cache_frame_data=False
            )
            
            plt.show()
            
        except KeyboardInterrupt:
            print("\n⏹️  Stopping...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop plotting and cleanup"""
        self.running = False
        
        if self.serial_conn:
            self.serial_conn.close()
            
        print(f"📈 Final Statistics:")
        print(f"  - Total samples: {self.sample_count:,}")
        duration = time.time() - self.start_time
        print(f"  - Duration: {duration:.1f} seconds")
        print(f"  - Average rate: {self.sample_count/duration:.1f} Hz")
        
        if self.ppg_data:
            print(f"  - PPG range: {min(self.ppg_data):.3f} to {max(self.ppg_data):.3f}")
        if self.hr_data:
            print(f"  - Heart rate: {np.mean(list(self.hr_data)):.1f} ± {np.std(list(self.hr_data)):.1f} BPM")
    
    def save_data(self, filename=None):
        """Save collected data to CSV file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arduino_ppg_data_{timestamp}.csv"
        
        if len(self.timestamps) > 0:
            df = pd.DataFrame({
                'timestamp': list(self.timestamps),
                'ecg': list(self.ecg_data),
                'ppg': list(self.ppg_data),
                'heart_rate': list(self.hr_data)
            })
            df.to_csv(filename, index=False)
            print(f"💾 Data saved to {filename}")

def find_arduino_ports():
    """Find potential Arduino ports"""
    import glob
    
    if sys.platform.startswith('darwin'):  # macOS
        ports = glob.glob('/dev/tty.usbmodem*') + glob.glob('/dev/tty.usbserial*') + glob.glob('/dev/cu.usbmodem*')
    elif sys.platform.startswith('linux'):  # Linux
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    elif sys.platform.startswith('win'):  # Windows
        ports = [f'COM{i}' for i in range(1, 20)]
    else:
        ports = []
    
    return ports

def main():
    parser = argparse.ArgumentParser(description='Real-time Arduino PPG Plotter')
    parser.add_argument('--port', help='Arduino serial port (auto-detected if not specified)')
    parser.add_argument('--baudrate', type=int, default=9600, help='Serial baudrate (default: 9600)')
    parser.add_argument('--window', type=int, default=1000, help='Data window size (default: 1000)')
    parser.add_argument('--list-ports', action='store_true', help='List available ports')
    parser.add_argument('--save', action='store_true', help='Save data to CSV file on exit')
    
    args = parser.parse_args()
    
    # List ports if requested
    if args.list_ports:
        ports = find_arduino_ports()
        print("🔌 Available ports:")
        for port in ports:
            print(f"  - {port}")
        return
    
    # Auto-detect port if not specified
    if not args.port:
        ports = find_arduino_ports()
        if not ports:
            print("❌ No Arduino ports found!")
            print("💡 Try: python3 arduino_ppg_realtime.py --list-ports")
            return
        args.port = ports[0]
        print(f"🔍 Auto-detected port: {args.port}")
    
    print("🚀 Arduino PPG Real-time Plotter")
    print("=" * 40)
    print(f"📡 Port: {args.port}")
    print(f"⚡ Baudrate: {args.baudrate}")
    print(f"🪟 Window size: {args.window}")
    print()
    
    # Create and start plotter
    plotter = ArduinoPPGPlotter(args.port, args.baudrate, args.window)
    
    try:
        plotter.start_plotting()
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    finally:
        if args.save:
            plotter.save_data()
        plotter.stop()

if __name__ == "__main__":
    main()
