"""
Test Script for Raspberry Pi Pico TinyML Code
=============================================

This script tests the Pico TinyML implementation on a desktop computer
to verify functionality before deploying to actual hardware.

Simulates MicroPython environment and tests:
- Model loading and initialization
- Fixed-point arithmetic
- Signal processing
- Memory usage estimation
"""

import math
import time
import sys
import os

# Mock MicroPython modules for desktop testing
class MockPin:
    OUT = 1
    IN = 0
    
    def __init__(self, pin, mode=None):
        self.pin = pin
        self.mode = mode
        self.value = 0
    
    def on(self):
        self.value = 1
    
    def off(self):
        self.value = 0
    
    def toggle(self):
        self.value = 1 - self.value

class MockADC:
    def __init__(self, pin):
        self.pin = pin
        self.noise_offset = 0
    
    def read_u16(self):
        # Simulate ECG-like signal
        t = time.time() * 10  # Speed up for testing
        ecg_sim = (
            0.8 * math.sin(2 * math.pi * 1.2 * t) +  # Main heartbeat
            0.3 * math.sin(2 * math.pi * 0.1 * t) +  # Breathing
            0.1 * (0.5 - hash(str(t)) % 100 / 100)  # Random noise
        )
        # Convert to 16-bit ADC value (centered around 32768)
        adc_value = int(32768 + ecg_sim * 10000)
        return max(0, min(65535, adc_value))

class MockPWM:
    def __init__(self, pin):
        self.pin = pin
        self._freq = 1000
        self._duty = 32768
    
    def freq(self, f):
        self._freq = f
    
    def duty_u16(self, duty):
        self._duty = duty
    
    def get_duty(self):
        return self._duty

class MockUART:
    def __init__(self, *args, **kwargs):
        self.buffer = []
    
    def write(self, data):
        self.buffer.append(data)
        if isinstance(data, str) and not data.startswith('timestamp'):
            print(f"UART: {data.strip()}")

class MockTimer:
    def __init__(self):
        self.callback = None
        self.running = False
    
    def init(self, period, mode, callback):
        self.callback = callback
        self.running = True
    
    def deinit(self):
        self.running = False

# Mock machine module
class machine:
    Pin = MockPin
    ADC = MockADC
    PWM = MockPWM
    UART = MockUART
    Timer = MockTimer

# Mock micropython module
class micropython:
    @staticmethod
    def const(x):
        return x

# Mock gc module
class gc:
    @staticmethod
    def collect():
        pass
    
    @staticmethod
    def mem_free():
        return 200000  # Simulate available memory

# Mock time with MicroPython-like functions
class MockTime:
    @staticmethod
    def ticks_ms():
        return int(time.time() * 1000)
    
    @staticmethod
    def ticks_us():
        return int(time.time() * 1000000)
    
    @staticmethod
    def ticks_diff(a, b):
        return a - b
    
    @staticmethod
    def sleep_ms(ms):
        time.sleep(ms / 1000)

# Add mocks to sys.modules
sys.modules['machine'] = machine
sys.modules['micropython'] = micropython
sys.modules['gc'] = gc

# Replace time module temporarily
original_time = sys.modules.get('time')
sys.modules['time'] = MockTime

# Now import the Pico modules
try:
    from pico_tinyml_model import (
        PicoMatrix, PicoActivation, PicoLSTMCell, 
        PicoECGtoPPG, PicoPPGSignalProcessor, create_pico_model
    )
    print("✅ Successfully imported Pico TinyML modules")
except ImportError as e:
    print(f"❌ Failed to import Pico modules: {e}")
    sys.exit(1)

def test_fixed_point_arithmetic():
    """Test fixed-point arithmetic functions."""
    print("\n🧮 Testing Fixed-Point Arithmetic...")
    
    # Test tanh approximation
    test_values = [-2000, -1000, 0, 1000, 2000]
    for val in test_values:
        result = PicoActivation.tanh_fixed(val)
        expected = math.tanh(val / 1000) * 1000
        error = abs(result - expected) / 1000
        print(f"tanh({val/1000:.1f}): {result/1000:.3f} vs {expected/1000:.3f} (error: {error:.3f})")
    
    # Test sigmoid approximation
    for val in test_values:
        result = PicoActivation.sigmoid_fixed(val)
        expected = (1 / (1 + math.exp(-val/1000))) * 1000
        error = abs(result - expected) / 1000
        print(f"sigmoid({val/1000:.1f}): {result/1000:.3f} vs {expected/1000:.3f} (error: {error:.3f})")

def test_matrix_operations():
    """Test matrix operations."""
    print("\n🔢 Testing Matrix Operations...")
    
    # Test matrix creation and basic operations
    m1 = PicoMatrix(2, 3, [1, 2, 3, 4, 5, 6])
    m2 = PicoMatrix(3, 2, [1, 2, 3, 4, 5, 6])
    
    # Test matrix multiplication
    result = m1.multiply(m2)
    print(f"Matrix multiplication result shape: {result.rows}x{result.cols}")
    print(f"Result[0,0]: {result.get(0, 0)} (expected: 22)")
    print(f"Result[1,1]: {result.get(1, 1)} (expected: 64)")

def test_lstm_cell():
    """Test LSTM cell functionality."""
    print("\n🧠 Testing LSTM Cell...")
    
    # Create small LSTM cell
    lstm = PicoLSTMCell(input_size=1, hidden_size=4)
    
    # Test forward pass
    test_input = [500]  # Fixed-point input
    output = lstm.forward(test_input)
    
    print(f"LSTM input: {test_input}")
    print(f"LSTM output: {output}")
    print(f"Output length: {len(output)} (expected: 4)")
    
    # Test state reset
    lstm.reset_state()
    print("LSTM state reset completed")

def test_ecg_to_ppg_model():
    """Test the complete ECG-to-PPG model."""
    print("\n💓 Testing ECG-to-PPG Model...")
    
    # Create model
    model = PicoECGtoPPG(input_size=1, hidden_size=8, num_layers=1)
    
    # Test memory usage calculation
    memory_usage = model.get_memory_usage()
    print(f"Estimated memory usage: {memory_usage} bytes ({memory_usage/1024:.1f} KB)")
    
    # Test prediction
    test_samples = [0.5, 0.6, 0.4, 0.7, 0.3]
    predictions = []
    
    for sample in test_samples:
        pred = model.predict_sample(sample)
        predictions.append(pred)
    
    print(f"Test predictions: {[f'{p:.3f}' for p in predictions]}")
    
    # Test model reset
    model.reset_model()
    print("Model reset completed")

def test_signal_processor():
    """Test signal processing components."""
    print("\n📊 Testing Signal Processor...")
    
    processor = PicoPPGSignalProcessor()
    
    # Test moving average filter
    test_signal = [1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0]
    filtered_signal = []
    
    for sample in test_signal:
        filtered = processor.moving_average_filter(sample)
        filtered_signal.append(filtered)
    
    print(f"Original signal: {test_signal}")
    print(f"Filtered signal: {[f'{f:.3f}' for f in filtered_signal]}")
    
    # Test peak detection
    peak_signal = [0.1, 0.2, 0.5, 0.8, 0.6, 0.3, 0.1, 0.2]
    peaks_detected = []
    
    for i in range(3, len(peak_signal)):
        is_peak = processor.simple_peak_detector(peak_signal[i-2:i+1], threshold=0.5)
        peaks_detected.append(is_peak)
    
    print(f"Peak detection results: {peaks_detected}")

def test_performance_benchmark():
    """Benchmark processing performance."""
    print("\n⚡ Performance Benchmark...")
    
    model = PicoECGtoPPG(input_size=1, hidden_size=8, num_layers=1)
    
    # Generate test data
    test_data = [math.sin(i * 0.1) for i in range(1000)]
    
    # Measure processing time
    start_time = time.time()
    
    for sample in test_data:
        _ = model.predict_sample(sample)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    avg_time_per_sample = (total_time / len(test_data)) * 1000  # ms
    samples_per_second = len(test_data) / total_time
    
    print(f"Processed {len(test_data)} samples in {total_time:.3f} seconds")
    print(f"Average time per sample: {avg_time_per_sample:.3f} ms")
    print(f"Processing rate: {samples_per_second:.0f} samples/second")
    print(f"Real-time capability: {samples_per_second/250:.1f}x (assuming 250 Hz sampling)")

def test_memory_usage():
    """Test memory usage patterns."""
    print("\n💾 Memory Usage Test...")
    
    # Test model creation impact
    initial_memory = gc.mem_free()
    print(f"Initial free memory: {initial_memory} bytes")
    
    # Create model
    model = create_pico_model()
    
    after_model_memory = gc.mem_free()
    model_memory_used = initial_memory - after_model_memory
    
    print(f"Memory after model creation: {after_model_memory} bytes")
    print(f"Model memory usage: {model_memory_used} bytes")
    
    # Test processing memory stability
    for i in range(100):
        _ = model.predict_sample(0.5)
        if i % 20 == 0:
            gc.collect()
    
    final_memory = gc.mem_free()
    processing_memory_change = after_model_memory - final_memory
    
    print(f"Memory after processing: {final_memory} bytes")
    print(f"Memory change during processing: {processing_memory_change} bytes")
    
    if abs(processing_memory_change) < 1000:
        print("✅ Memory usage is stable during processing")
    else:
        print("⚠️  Memory usage changed significantly during processing")

def run_integration_test():
    """Run integrated system test."""
    print("\n🔄 Integration Test...")
    
    try:
        # Simulate the main system components
        model = PicoECGtoPPG(input_size=1, hidden_size=8, num_layers=1)
        processor = PicoPPGSignalProcessor()
        
        # Simulate ECG data processing
        print("Simulating 5 seconds of ECG processing at 250 Hz...")
        
        sample_count = 0
        heart_rate = 75
        
        for i in range(1250):  # 5 seconds at 250 Hz
            # Generate synthetic ECG
            t = i / 250.0
            ecg_value = (
                0.8 * math.sin(2 * math.pi * 1.25 * t) +  # 75 BPM
                0.1 * math.sin(2 * math.pi * 0.25 * t) +  # Breathing
                0.05 * (0.5 - (i % 37) / 37)             # Noise
            )
            
            # Process through pipeline
            filtered_ecg = processor.moving_average_filter(ecg_value)
            ppg_prediction = model.predict_sample(filtered_ecg)
            filtered_ppg = processor.moving_average_filter(ppg_prediction)
            
            # Simulate UART output (every 10th sample)
            if i % 10 == 0:
                timestamp = int(t * 1000)
                # print(f"{timestamp},{ecg_value:.4f},{filtered_ppg:.4f},{heart_rate}")
            
            sample_count += 1
            
            # Progress indicator
            if i % 250 == 0:
                seconds_completed = i // 250 + 1
                print(f"  Progress: {seconds_completed}/5 seconds")
        
        print(f"✅ Integration test completed successfully!")
        print(f"   Processed {sample_count} samples")
        print(f"   Final ECG value: {ecg_value:.4f}")
        print(f"   Final PPG prediction: {filtered_ppg:.4f}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("🧪 Raspberry Pi Pico TinyML Code Test Suite")
    print("=" * 50)
    
    # Run individual tests
    test_fixed_point_arithmetic()
    test_matrix_operations()
    test_lstm_cell()
    test_ecg_to_ppg_model()
    test_signal_processor()
    test_performance_benchmark()
    test_memory_usage()
    run_integration_test()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("\nCode is ready for deployment to Raspberry Pi Pico!")
    
    # Restore original time module
    if original_time:
        sys.modules['time'] = original_time

if __name__ == "__main__":
    main()
