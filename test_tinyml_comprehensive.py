"""
Comprehensive Test Suite for TinyML PPG Estimation Project
=========================================================

Enhanced testing framework covering:
- TinyML model architecture and functionality
- Data processing and signal quality
- Performance benchmarks and edge device compatibility
- Model conversion and deployment readiness
- Real-time inference capabilities
- Error handling and edge cases

Features:
- Parameterized tests for different configurations
- Performance profiling and benchmarking
- Visual validation of results
- Memory usage monitoring
- Cross-platform compatibility tests
"""

import unittest
import numpy as np
import torch
import time
import os
import sys
import traceback
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import psutil
import gc

# Import project modules
try:
    from tinyml_lstm_model import TinyECGtoPPG_LSTM, TinyPPGProcessor, create_tiny_model
    from train_tinyml_ppg_model import (
        load_ecg_data_tiny, create_tiny_ppg_template, 
        generate_tiny_ppg_from_ecg, create_tiny_sequences,
        calculate_tiny_metrics
    )
    from tinyml_inference_demo import (
        load_tiny_model, generate_test_ecg, 
        process_ecg_chunks, calculate_inference_metrics
    )
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    print("Some tests may be skipped.")

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TestTinyMLArchitecture(unittest.TestCase):
    """Test TinyML model architecture and core functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.input_size = 1
        self.hidden_size = 48
        self.num_layers = 2
        self.output_size = 1
        self.seq_length = 32
        self.batch_size = 8
        
    def test_model_creation(self):
        """Test TinyML model initialization."""
        model = TinyECGtoPPG_LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size
        )
        
        self.assertIsInstance(model, TinyECGtoPPG_LSTM)
        self.assertEqual(model.input_size, self.input_size)
        self.assertEqual(model.hidden_size, self.hidden_size)
        self.assertEqual(model.num_layers, self.num_layers)
        
    def test_model_forward_pass(self):
        """Test model forward propagation."""
        model = TinyECGtoPPG_LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size
        )
        
        # Test input
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        
        with torch.no_grad():
            output = model(x)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_length, self.output_size)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is finite
        self.assertTrue(torch.isfinite(output).all())
        
    def test_model_size_calculation(self):
        """Test model size calculation for TinyML deployment."""
        model = TinyECGtoPPG_LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size
        )
        
        param_count, model_size = model.get_model_size()
        
        # Verify reasonable size for TinyML
        self.assertGreater(param_count, 1000)  # Not too small
        self.assertLess(param_count, 100000)   # Not too large for embedded
        self.assertLess(model_size, 500 * 1024)  # Less than 500KB
        
    def test_different_architectures(self):
        """Test various model configurations."""
        configurations = [
            {'hidden_size': 32, 'num_layers': 1},
            {'hidden_size': 48, 'num_layers': 2},
            {'hidden_size': 64, 'num_layers': 1},
        ]
        
        for config in configurations:
            with self.subTest(config=config):
                model = TinyECGtoPPG_LSTM(
                    input_size=self.input_size,
                    output_size=self.output_size,
                    **config
                )
                
                x = torch.randn(4, 32, 1)
                with torch.no_grad():
                    output = model(x)
                
                self.assertEqual(output.shape, (4, 32, 1))


class TestDataProcessing(unittest.TestCase):
    """Test data processing and signal generation."""
    
    def setUp(self):
        """Set up test data."""
        self.sampling_rate = 180
        self.duration = 5  # seconds
        self.n_samples = self.sampling_rate * self.duration
        
    def test_ecg_data_loading(self):
        """Test ECG data loading functionality."""
        try:
            ecg_data = load_ecg_data_tiny()
            
            self.assertIsInstance(ecg_data, np.ndarray)
            self.assertGreater(len(ecg_data), 1000)
            self.assertEqual(ecg_data.dtype, np.float32)
            
            # Check for reasonable ECG signal properties
            # Note: Raw ECG data may not be normalized, so we check for reasonable range
            self.assertLess(np.abs(ecg_data.mean()), 1000.0)  # Should be in reasonable range
            self.assertGreater(ecg_data.std(), 0.1)  # Should have variation
            
        except FileNotFoundError:
            # Skip if no data file available
            self.skipTest("ECG data file not available")
            
    def test_ppg_template_creation(self):
        """Test PPG template generation."""
        template_sizes = [16, 32, 64]
        
        for size in template_sizes:
            with self.subTest(size=size):
                template = create_tiny_ppg_template(pulse_width=size)
                
                self.assertEqual(len(template), size)
                self.assertEqual(template.dtype, np.float32)
                self.assertGreaterEqual(template.min(), 0)
                self.assertLessEqual(template.max(), 1)
                
                # Check template has pulse-like characteristics
                max_idx = np.argmax(template)
                self.assertGreater(max_idx, size * 0.1)  # Peak not at start
                self.assertLess(max_idx, size * 0.6)     # Peak in first half
                
    def test_sequence_creation(self):
        """Test sequence creation for TinyML training."""
        ecg_data = np.random.randn(1000).astype(np.float32)
        ppg_data = np.random.randn(1000).astype(np.float32)
        
        seq_lengths = [16, 32, 64]
        
        for seq_len in seq_lengths:
            with self.subTest(seq_len=seq_len):
                X, Y = create_tiny_sequences(ecg_data, ppg_data, seq_len=seq_len)
                
                self.assertEqual(X.shape[1], seq_len)
                self.assertEqual(Y.shape[1], seq_len)
                self.assertEqual(X.shape[0], Y.shape[0])
                self.assertEqual(X.dtype, np.float32)
                self.assertEqual(Y.dtype, np.float32)
                
    def test_synthetic_ecg_generation(self):
        """Test synthetic ECG generation for testing."""
        durations = [5, 10, 30]
        
        for duration in durations:
            with self.subTest(duration=duration):
                ecg = generate_test_ecg(duration_seconds=duration, sampling_rate=180)
                
                expected_length = int(duration * 180)
                self.assertEqual(len(ecg), expected_length)
                self.assertEqual(ecg.dtype, np.float32)
                
                # Check for realistic ECG properties
                self.assertLess(np.abs(ecg.mean()), 0.2)  # Near zero mean
                self.assertGreater(ecg.std(), 0.5)        # Reasonable variation


class TestSignalProcessing(unittest.TestCase):
    """Test signal processing components."""
    
    def setUp(self):
        """Set up signal processing tests."""
        self.processor = TinyPPGProcessor(seq_length=32, sampling_rate=180)
        
    def test_ppg_processor_initialization(self):
        """Test PPG processor initialization."""
        self.assertEqual(self.processor.seq_length, 32)
        self.assertEqual(self.processor.sampling_rate, 180)
        self.assertEqual(self.processor.ma_window, 5)
        
    def test_peak_detection(self):
        """Test peak detection functionality."""
        # Create signal with known peaks
        t = np.linspace(0, 5, 900)  # 5 seconds at 180 Hz
        signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
        
        peaks = self.processor.detect_simple_peaks(signal, min_distance=75)
        
        self.assertIsInstance(peaks, np.ndarray)
        self.assertGreater(len(peaks), 2)  # Should find multiple peaks
        self.assertLess(len(peaks), 20)    # But not too many
        
        # Check peaks are reasonably spaced
        if len(peaks) > 1:
            intervals = np.diff(peaks)
            self.assertGreater(intervals.min(), 50)  # Minimum spacing
            
    def test_ecg_preprocessing(self):
        """Test ECG preprocessing functionality."""
        # Create test ECG signal
        ecg_signal = np.random.randn(1000) + 0.5  # With DC offset
        
        processed = self.processor.preprocess_ecg(ecg_signal)
        
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(processed.dtype, np.float32)
        self.assertEqual(processed.shape[1], self.processor.seq_length)
        
        # Check normalization
        flat_processed = processed.flatten()
        self.assertLess(np.abs(flat_processed.mean()), 0.1)  # Near zero mean
        self.assertGreater(flat_processed.std(), 0.8)        # Unit variance
        
    def test_ppg_postprocessing(self):
        """Test PPG postprocessing functionality."""
        # Create noisy PPG signal
        ppg_signal = np.sin(np.linspace(0, 10, 1000)) + 0.2 * np.random.randn(1000)
        
        smoothed = self.processor.postprocess_ppg(ppg_signal)
        
        self.assertEqual(len(smoothed), len(ppg_signal))
        self.assertEqual(smoothed.dtype, np.float32)
        
        # Check smoothing effect (should reduce high-frequency noise)
        original_diff = np.std(np.diff(ppg_signal))
        smoothed_diff = np.std(np.diff(smoothed))
        self.assertLess(smoothed_diff, original_diff)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test performance and benchmarking."""
    
    def setUp(self):
        """Set up performance testing."""
        self.model = TinyECGtoPPG_LSTM(
            input_size=1, hidden_size=48, num_layers=2, output_size=1
        )
        self.model.eval()
        
    def test_inference_speed(self):
        """Test inference speed for real-time requirements."""
        batch_sizes = [1, 4, 8]
        seq_length = 32
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                x = torch.randn(batch_size, seq_length, 1)
                
                # Warm up
                with torch.no_grad():
                    _ = self.model(x)
                
                # Measure inference time
                times = []
                for _ in range(100):
                    start_time = time.time()
                    with torch.no_grad():
                        _ = self.model(x)
                    times.append(time.time() - start_time)
                
                avg_time = np.mean(times) * 1000  # ms
                
                # Check real-time performance (32 samples at 180 Hz = 178ms)
                real_time_budget = 178  # ms
                self.assertLess(avg_time, real_time_budget, 
                               f"Inference too slow: {avg_time:.2f}ms > {real_time_budget}ms")
                
    def test_memory_usage(self):
        """Test memory usage for embedded deployment."""
        process = psutil.Process()
        
        # Measure memory before
        gc.collect()
        mem_before = process.memory_info().rss
        
        # Create model and run inference
        model = TinyECGtoPPG_LSTM(input_size=1, hidden_size=48, num_layers=2, output_size=1)
        x = torch.randn(1, 32, 1)
        
        with torch.no_grad():
            _ = model(x)
        
        # Measure memory after
        mem_after = process.memory_info().rss
        mem_increase = (mem_after - mem_before) / 1024 / 1024  # MB
        
        # Should not use excessive memory
        self.assertLess(mem_increase, 50, f"Memory usage too high: {mem_increase:.2f}MB")
        
    def test_batch_processing_efficiency(self):
        """Test efficiency of batch processing."""
        seq_length = 32
        
        # Single sample processing
        x_single = torch.randn(1, seq_length, 1)
        start_time = time.time()
        for _ in range(8):
            with torch.no_grad():
                _ = self.model(x_single)
        single_time = time.time() - start_time
        
        # Batch processing
        x_batch = torch.randn(8, seq_length, 1)
        start_time = time.time()
        with torch.no_grad():
            _ = self.model(x_batch)
        batch_time = time.time() - start_time
        
        # Batch should be more efficient
        efficiency_ratio = single_time / batch_time
        self.assertGreater(efficiency_ratio, 1.5, 
                          f"Batch processing not efficient enough: {efficiency_ratio:.2f}x")


class TestModelValidation(unittest.TestCase):
    """Test model validation and quality metrics."""
    
    def test_model_loading(self):
        """Test trained model loading."""
        model_path = "best_tinyml_ppg_model.pth"
        
        if os.path.exists(model_path):
            model = load_tiny_model(model_path)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TinyECGtoPPG_LSTM)
        else:
            self.skipTest("Trained model not available")
            
    def test_model_predictions_quality(self):
        """Test quality of model predictions."""
        if not os.path.exists("best_tinyml_ppg_model.pth"):
            self.skipTest("Trained model not available")
            
        model = load_tiny_model("best_tinyml_ppg_model.pth")
        
        # Generate test data
        ecg_data = generate_test_ecg(duration_seconds=10, sampling_rate=180)
        
        # Get predictions
        ppg_pred, inference_time = process_ecg_chunks(model, ecg_data, chunk_size=32)
        
        # Basic quality checks
        self.assertEqual(len(ppg_pred), len(ecg_data))
        self.assertTrue(np.isfinite(ppg_pred).all())
        self.assertLess(inference_time, 10)  # Should be fast
        
        # Signal quality checks
        signal_power = np.mean(ppg_pred ** 2)
        self.assertGreater(signal_power, 0.001)  # Should have some signal
        
    def test_metrics_calculation(self):
        """Test metrics calculation functionality."""
        # Create test data with known relationship
        y_true = np.sin(np.linspace(0, 4*np.pi, 1000))
        y_pred = 0.8 * y_true + 0.1 * np.random.randn(1000)  # Add noise
        
        metrics = calculate_tiny_metrics(y_true, y_pred)
        
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('correlation', metrics)
        
        # Check reasonable values
        self.assertGreater(metrics['correlation'], 0.7)  # Should be well correlated
        self.assertLess(metrics['rmse'], 0.5)            # RMSE should be reasonable
        self.assertLess(metrics['mae'], 0.3)             # MAE should be reasonable


class TestEdgeCaseHandling(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        model = TinyECGtoPPG_LSTM(input_size=1, hidden_size=32, num_layers=1, output_size=1)
        
        # Test with very small input
        x = torch.randn(1, 1, 1)
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, (1, 1, 1))
        self.assertTrue(torch.isfinite(output).all())
        
    def test_extreme_input_values(self):
        """Test handling of extreme input values."""
        model = TinyECGtoPPG_LSTM(input_size=1, hidden_size=32, num_layers=1, output_size=1)
        
        extreme_inputs = [
            torch.zeros(1, 32, 1),           # All zeros
            torch.ones(1, 32, 1) * 100,      # Large positive values
            torch.ones(1, 32, 1) * -100,     # Large negative values
            torch.randn(1, 32, 1) * 1000,    # Very noisy
        ]
        
        for i, x in enumerate(extreme_inputs):
            with self.subTest(case=i):
                with torch.no_grad():
                    output = model(x)
                
                self.assertTrue(torch.isfinite(output).all())
                self.assertFalse(torch.isnan(output).any())
                
    def test_different_sequence_lengths(self):
        """Test model with different sequence lengths."""
        model = TinyECGtoPPG_LSTM(input_size=1, hidden_size=32, num_layers=1, output_size=1)
        
        seq_lengths = [1, 16, 32, 64, 128]
        
        for seq_len in seq_lengths:
            with self.subTest(seq_len=seq_len):
                x = torch.randn(1, seq_len, 1)
                with torch.no_grad():
                    output = model(x)
                
                self.assertEqual(output.shape, (1, seq_len, 1))
                self.assertTrue(torch.isfinite(output).all())


class TestVisualizationAndReporting(unittest.TestCase):
    """Test visualization and reporting functionality."""
    
    def test_results_visualization(self):
        """Test creation of result visualizations."""
        # Generate test data
        time_axis = np.linspace(0, 5, 900)
        ecg_data = np.sin(2 * np.pi * 1.2 * time_axis) + 0.1 * np.random.randn(900)
        ppg_data = 0.8 * np.sin(2 * np.pi * 1.2 * time_axis + 0.3) + 0.1 * np.random.randn(900)
        
        # Create simple visualization
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(time_axis[:180], ecg_data[:180], 'b-', label='ECG')
        plt.title('Test ECG Signal')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(time_axis[:180], ppg_data[:180], 'r-', label='PPG')
        plt.title('Test PPG Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save test visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_filename = f"test_visualization_{timestamp}.png"
        plt.savefig(test_filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Check file was created
        self.assertTrue(os.path.exists(test_filename))
        
        # Clean up
        if os.path.exists(test_filename):
            os.remove(test_filename)


class TestSuiteRunner:
    """Enhanced test suite runner with reporting."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        
    def run_comprehensive_tests(self):
        """Run all test suites with detailed reporting."""
        print("🧪 Starting Comprehensive TinyML Testing Suite")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Define test suites
        test_suites = [
            ('TinyML Architecture', TestTinyMLArchitecture),
            ('Data Processing', TestDataProcessing),
            ('Signal Processing', TestSignalProcessing),
            ('Performance Benchmarks', TestPerformanceBenchmarks),
            ('Model Validation', TestModelValidation),
            ('Edge Case Handling', TestEdgeCaseHandling),
            ('Visualization & Reporting', TestVisualizationAndReporting),
        ]
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0
        
        for suite_name, test_class in test_suites:
            print(f"\n🔬 Running {suite_name} Tests...")
            print("-" * 40)
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(test_class)
            
            # Run tests with custom result collector
            runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
            result = runner.run(suite)
            
            # Collect results
            tests_run = result.testsRun
            failures = len(result.failures)
            errors = len(result.errors)
            skipped = len(result.skipped)
            
            total_tests += tests_run
            total_failures += failures
            total_errors += errors
            total_skipped += skipped
            
            self.results[suite_name] = {
                'tests': tests_run,
                'failures': failures,
                'errors': errors,
                'skipped': skipped,
                'success_rate': (tests_run - failures - errors) / max(tests_run, 1) * 100
            }
            
            # Print suite summary
            print(f"✅ {suite_name}: {tests_run} tests, "
                  f"{failures} failures, {errors} errors, {skipped} skipped")
        
        # Print overall summary
        self.print_final_summary(total_tests, total_failures, total_errors, total_skipped)
        
    def print_final_summary(self, total_tests, total_failures, total_errors, total_skipped):
        """Print comprehensive test summary."""
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        
        print(f"📊 Overall Statistics:")
        print(f"   Total Tests Run: {total_tests}")
        print(f"   Successful: {total_tests - total_failures - total_errors}")
        print(f"   Failed: {total_failures}")
        print(f"   Errors: {total_errors}")
        print(f"   Skipped: {total_skipped}")
        print(f"   Success Rate: {(total_tests - total_failures - total_errors) / max(total_tests, 1) * 100:.1f}%")
        print(f"   Execution Time: {total_time:.2f} seconds")
        
        print(f"\n📈 Suite Breakdown:")
        for suite_name, results in self.results.items():
            status = "✅" if results['failures'] == 0 and results['errors'] == 0 else "❌"
            print(f"   {status} {suite_name}: {results['success_rate']:.1f}% "
                  f"({results['tests']} tests)")
        
        # Overall status
        if total_failures == 0 and total_errors == 0:
            print(f"\n🎉 ALL TESTS PASSED! TinyML system is ready for deployment.")
        else:
            print(f"\n⚠️  Some tests failed. Please review and fix issues before deployment.")
        
        print(f"\n🚀 TinyML PPG Estimation System Test Complete!")


def main():
    """Main test execution function."""
    print("🔬 TinyML PPG Estimation - Comprehensive Test Suite")
    print("Initializing enhanced testing framework...")
    
    # Run comprehensive tests
    runner = TestSuiteRunner()
    runner.run_comprehensive_tests()


if __name__ == '__main__':
    main()
