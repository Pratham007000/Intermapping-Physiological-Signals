"""
Raspberry Pi Pico Main Application
==================================

Real-time ECG-to-PPG conversion using TinyML on Raspberry Pi Pico.
Interfaces with ADC for ECG input and provides PPG output via PWM/UART.

Hardware Setup:
- ECG Input: GPIO 26 (ADC0) - Connect ECG sensor
- PPG Output: GPIO 15 (PWM) - Connect to LED/DAC for visualization
- Status LED: GPIO 25 (Built-in LED)
- UART: GPIO 0/1 for data output to computer

Usage:
1. Upload this file and pico_tinyml_model.py to Pico
2. Connect ECG sensor to ADC0 (GPIO 26)
3. Run the application
4. Monitor output via UART or PWM signal
"""

from machine import Pin, ADC, PWM, UART, Timer
import time
import gc
from pico_tinyml_model import create_pico_model, PicoPPGSignalProcessor

# Hardware Configuration
ECG_ADC_PIN = 26        # ADC0 for ECG input
PPG_PWM_PIN = 15        # PWM output for PPG signal
STATUS_LED_PIN = 25     # Built-in LED for status
UART_TX_PIN = 0         # UART TX for data output
UART_RX_PIN = 1         # UART RX

# Sampling Configuration
SAMPLING_RATE_HZ = 250  # ECG sampling rate (Hz)
SAMPLING_PERIOD_MS = int(1000 / SAMPLING_RATE_HZ)  # Sampling period in ms

# ADC Configuration
ADC_MAX_VALUE = 65535   # 16-bit ADC
ADC_REFERENCE_V = 3.3   # ADC reference voltage

class PicoECGPPGSystem:
    """Main system class for ECG-to-PPG conversion on Pico."""
    
    def __init__(self):
        print("Initializing Pico ECG-to-PPG System...")
        
        # Initialize hardware
        self.setup_hardware()
        
        # Initialize TinyML model
        self.setup_model()
        
        # Initialize signal processing
        self.setup_signal_processing()
        
        # System state
        self.is_running = False
        self.sample_count = 0
        self.heart_rate = 0
        self.last_peak_time = 0
        
        print("System initialized successfully!")
    
    def setup_hardware(self):
        """Initialize all hardware components."""
        print("Setting up hardware...")
        
        # ECG input (ADC)
        self.ecg_adc = ADC(Pin(ECG_ADC_PIN))
        
        # PPG output (PWM)
        self.ppg_pwm = PWM(Pin(PPG_PWM_PIN))
        self.ppg_pwm.freq(1000)  # 1kHz PWM frequency
        self.ppg_pwm.duty_u16(32768)  # Start at 50% duty cycle
        
        # Status LED
        self.status_led = Pin(STATUS_LED_PIN, Pin.OUT)
        
        # UART for data output
        self.uart = UART(0, baudrate=115200, tx=Pin(UART_TX_PIN), rx=Pin(UART_RX_PIN))
        
        # Timer for sampling
        self.sample_timer = Timer()
        
        print("Hardware setup complete!")
    
    def setup_model(self):
        """Initialize the TinyML model."""
        print("Loading TinyML model...")
        
        try:
            self.model = create_pico_model()
            self.model_loaded = True
            print("TinyML model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
    
    def setup_signal_processing(self):
        """Initialize signal processing components."""
        self.signal_processor = PicoPPGSignalProcessor()
        
        # Signal buffers
        self.ecg_buffer = [0] * 20  # ECG sample buffer for peak detection
        self.ppg_buffer = [0] * 20  # PPG output buffer
        self.buffer_index = 0
        
        # Peak detection
        self.peak_detection_buffer = [0] * 10
        self.last_peaks = []
        self.peak_threshold = 0.3
    
    def read_ecg_sample(self):
        """Read ECG sample from ADC."""
        # Read ADC value
        adc_value = self.ecg_adc.read_u16()
        
        # Convert to voltage (0-3.3V)
        voltage = (adc_value / ADC_MAX_VALUE) * ADC_REFERENCE_V
        
        # Convert to normalized ECG value (-1 to 1)
        # Assuming ECG signal is centered around 1.65V with ±1V range
        ecg_normalized = (voltage - 1.65) / 1.0
        
        # Clamp to valid range
        ecg_normalized = max(-1.0, min(1.0, ecg_normalized))
        
        return ecg_normalized
    
    def process_sample(self, timer=None):
        """Process a single ECG sample and generate PPG output."""
        if not self.model_loaded:
            return
        
        try:
            # Read ECG sample
            ecg_sample = self.read_ecg_sample()
            
            # Add to ECG buffer
            self.ecg_buffer[self.buffer_index] = ecg_sample
            self.buffer_index = (self.buffer_index + 1) % len(self.ecg_buffer)\n            
            # Apply signal processing (optional filtering)
            filtered_ecg = self.signal_processor.moving_average_filter(ecg_sample)
            
            # Generate PPG prediction using TinyML model
            ppg_prediction = self.model.predict_sample(filtered_ecg)
            
            # Apply post-processing
            filtered_ppg = self.signal_processor.moving_average_filter(ppg_prediction)
            
            # Add to PPG buffer
            ppg_index = self.sample_count % len(self.ppg_buffer)
            self.ppg_buffer[ppg_index] = filtered_ppg
            
            # Update PWM output (convert PPG to PWM duty cycle)
            pwm_duty = self.ppg_to_pwm(filtered_ppg)
            self.ppg_pwm.duty_u16(pwm_duty)
            
            # Peak detection for heart rate estimation
            if self.signal_processor.simple_peak_detector(self.ppg_buffer[-3:], self.peak_threshold):
                current_time = time.ticks_ms()
                if self.last_peak_time > 0:
                    peak_interval = time.ticks_diff(current_time, self.last_peak_time)
                    if peak_interval > 300:  # Minimum 300ms between peaks (200 BPM max)
                        self.heart_rate = int(60000 / peak_interval)  # Convert to BPM
                        self.last_peaks.append(current_time)
                        if len(self.last_peaks) > 5:
                            self.last_peaks.pop(0)
                self.last_peak_time = current_time
            
            # Send data via UART (every 10th sample to reduce bandwidth)
            if self.sample_count % 10 == 0:
                self.send_uart_data(ecg_sample, filtered_ppg, self.heart_rate)
            
            self.sample_count += 1
            
            # Blink status LED every second
            if self.sample_count % SAMPLING_RATE_HZ == 0:
                self.status_led.toggle()
            
        except Exception as e:
            print(f"Error in sample processing: {e}")
    
    def ppg_to_pwm(self, ppg_value):
        """Convert PPG value to PWM duty cycle."""
        # Normalize PPG value to 0-1 range
        normalized = (ppg_value + 1.0) / 2.0  # Assuming PPG is in -1 to 1 range
        normalized = max(0.0, min(1.0, normalized))  # Clamp to valid range
        
        # Convert to 16-bit PWM duty cycle (0-65535)
        pwm_duty = int(normalized * 65535)
        
        return pwm_duty
    
    def send_uart_data(self, ecg, ppg, hr):
        """Send data via UART in CSV format."""
        try:
            timestamp = time.ticks_ms()
            data_line = f"{timestamp},{ecg:.4f},{ppg:.4f},{hr}\\n"
            self.uart.write(data_line)
        except Exception as e:
            print(f"UART error: {e}")
    
    def start_sampling(self):
        """Start real-time ECG sampling and PPG generation."""
        if self.is_running:
            print("System already running!")
            return
        
        print(f"Starting real-time sampling at {SAMPLING_RATE_HZ} Hz...")
        print("Press Ctrl+C to stop")
        
        self.is_running = True
        self.sample_count = 0
        
        # Start sampling timer
        self.sample_timer.init(
            period=SAMPLING_PERIOD_MS,
            mode=Timer.PERIODIC,
            callback=self.process_sample
        )
        
        # Send UART header
        self.uart.write("timestamp,ecg,ppg,heart_rate\\n")
        
        print("System running! ECG->PPG conversion active.")
    
    def stop_sampling(self):
        """Stop sampling and cleanup."""
        if not self.is_running:
            return
        
        print("Stopping system...")
        
        self.is_running = False
        self.sample_timer.deinit()
        
        # Reset PWM to neutral
        self.ppg_pwm.duty_u16(32768)
        
        # Turn off status LED
        self.status_led.off()
        
        print("System stopped.")
    
    def run_demo_mode(self, duration_seconds=30):
        """Run a demo with synthetic ECG data."""
        print(f"Running demo mode for {duration_seconds} seconds...")
        
        if not self.model_loaded:
            print("Model not loaded! Cannot run demo.")
            return
        
        demo_samples = duration_seconds * SAMPLING_RATE_HZ
        
        for i in range(demo_samples):
            # Generate synthetic ECG (simulated heartbeat)
            t = i / SAMPLING_RATE_HZ
            ecg_synthetic = (
                0.8 * self.synthetic_qrs(t, 75) +  # 75 BPM heartbeat
                0.1 * (2 * (i % 50) / 50 - 1) +    # Slow drift
                0.05 * (2 * (i % 5) / 5 - 1)       # High frequency noise
            )
            
            # Process through model
            ppg_prediction = self.model.predict_sample(ecg_synthetic)
            
            # Update PWM output
            pwm_duty = self.ppg_to_pwm(ppg_prediction)
            self.ppg_pwm.duty_u16(pwm_duty)
            
            # Send UART data
            if i % 10 == 0:
                self.send_uart_data(ecg_synthetic, ppg_prediction, 75)
            
            # Status LED blink
            if i % SAMPLING_RATE_HZ == 0:
                self.status_led.toggle()
                print(f"Demo progress: {i // SAMPLING_RATE_HZ + 1}/{duration_seconds} seconds")
            
            # Maintain sampling rate
            time.sleep_ms(SAMPLING_PERIOD_MS)
        
        print("Demo completed!")
    
    def synthetic_qrs(self, t, bpm):
        """Generate synthetic QRS complex."""
        import math
        
        # QRS occurs every 60/bpm seconds
        qrs_period = 60.0 / bpm
        
        # Time within current cardiac cycle
        cycle_time = t % qrs_period
        
        # QRS complex shape (simplified)
        if cycle_time < 0.1:  # QRS duration ~100ms
            # Simple triangular QRS
            if cycle_time < 0.05:
                return cycle_time / 0.05  # Rising edge
            else:
                return (0.1 - cycle_time) / 0.05  # Falling edge
        else:
            return 0.1 * math.sin(2 * math.pi * cycle_time / qrs_period)  # T-wave
    
    def get_system_status(self):
        """Get current system status."""
        gc.collect()
        free_memory = gc.mem_free()
        
        status = {
            'running': self.is_running,
            'sample_count': self.sample_count,
            'heart_rate': self.heart_rate,
            'free_memory': free_memory,
            'model_loaded': self.model_loaded
        }
        
        return status
    
    def print_status(self):
        """Print current system status."""
        status = self.get_system_status()
        
        print("=== System Status ===")
        print(f"Running: {status['running']}")
        print(f"Samples processed: {status['sample_count']}")
        print(f"Heart rate: {status['heart_rate']} BPM")
        print(f"Free memory: {status['free_memory']} bytes")
        print(f"Model loaded: {status['model_loaded']}")
        print("====================")


def main():
    """Main application entry point."""
    print("Raspberry Pi Pico ECG-to-PPG TinyML System")
    print("=" * 50)
    
    try:
        # Create system
        system = PicoECGPPGSystem()
        
        # Print initial status
        system.print_status()
        
        print("\\nOptions:")
        print("1. Run real-time mode (requires ECG sensor)")
        print("2. Run demo mode (synthetic ECG)")
        print("3. Show status")
        
        # For automatic demo, uncomment the next line:
        system.run_demo_mode(duration_seconds=30)
        
        # For interactive mode, uncomment the following:
        """
        while True:
            print("\\nEnter command (1=real-time, 2=demo, 3=status, q=quit):")
            # In real Pico, you'd read from UART or use buttons
            # For demo, we'll run demo mode
            system.run_demo_mode(10)
            break
        """
        
    except KeyboardInterrupt:
        print("\\nInterrupted by user")
        if 'system' in locals():
            system.stop_sampling()
    except Exception as e:
        print(f"System error: {e}")
        import sys
        sys.print_exception(e)
    
    print("Application terminated.")


if __name__ == "__main__":
    main()
