#!/usr/bin/env python3
"""
Comprehensive Test Suite for PPG Estimation Project
===================================================

This script contains unit tests for various components of the PPG estimation project,
including data loading, signal processing, model functionality, and performance metrics.

Usage:
    python test_ppg_estimation_comprehensive.py
    python -m unittest test_ppg_estimation_comprehensive.py
    python -m pytest test_ppg_estimation_comprehensive.py (if pytest is installed)
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import tempfile
import warnings
from unittest.mock import patch, MagicMock
import pandas as pd

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import project modules with error handling
try:
    from lstm_ppg_model import ECGtoPPG_LSTM
    LSTM_MODEL_AVAILABLE = True
except ImportError:
    LSTM_MODEL_AVAILABLE = False
    print("Warning: LSTM model not available for testing")

try:
    from train_improved_ppg_model import generate_ppg_from_ecg, create_realistic_ppg_template
    TRAINING_MODULE_AVAILABLE = True
except ImportError:
    TRAINING_MODULE_AVAILABLE = False
    print("Warning: Training module not available for testing")

try:
    from ecg_to_ppg_testing import create_sequences, calculate_metrics
    TESTING_MODULE_AVAILABLE = True
except ImportError:
    TESTING_MODULE_AVAILABLE = False
    print("Warning: Testing module not available for testing")


class TestDataGeneration(unittest.TestCase):
    """Test cases for data generation and signal processing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_ecg = np.sin(np.linspace(0, 10*np.pi, 1000)) + 0.1*np.random.randn(1000)
        self.sample_size = 1000
    
    @unittest.skipUnless(TRAINING_MODULE_AVAILABLE, "Training module not available")
    def test_generate_ppg_from_ecg_basic(self):
        """Test basic PPG generation from ECG signal."""
        ppg_signal = generate_ppg_from_ecg(self.sample_ecg)
        
        # Check output properties
        self.assertIsInstance(ppg_signal, np.ndarray)
        self.assertEqual(len(ppg_signal), len(self.sample_ecg))
        self.assertTrue(np.isfinite(ppg_signal).all(), "PPG signal contains non-finite values")
    
    @unittest.skipUnless(TRAINING_MODULE_AVAILABLE, "Training module not available")
    def test_generate_ppg_with_custom_parameters(self):
        """Test PPG generation with custom delay and pulse width."""
        delay = 150
        pulse_width = 100
        ppg_signal = generate_ppg_from_ecg(self.sample_ecg, delay=delay, pulse_width=pulse_width)
        
        self.assertIsInstance(ppg_signal, np.ndarray)
        self.assertEqual(len(ppg_signal), len(self.sample_ecg))
    
    @unittest.skipUnless(TRAINING_MODULE_AVAILABLE, "Training module not available")
    def test_create_realistic_ppg_template(self):
        """Test PPG template creation."""
        template = create_realistic_ppg_template(pulse_width=120)
        
        self.assertIsInstance(template, np.ndarray)
        self.assertGreater(len(template), 0)
        self.assertTrue(np.isfinite(template).all(), "Template contains non-finite values")
        self.assertLessEqual(np.max(template), 1.0, "Template not properly normalized")
    
    def test_generate_ppg_edge_cases(self):
        """Test PPG generation with edge cases."""
        if not TRAINING_MODULE_AVAILABLE:
            self.skipTest("Training module not available")
        
        # Empty signal - should handle gracefully and return empty array
        empty_signal = np.array([])
        ppg_empty = generate_ppg_from_ecg(empty_signal)
        self.assertEqual(len(ppg_empty), 0)
        
        # Very short signal
        short_signal = np.array([1, 2, 3])
        ppg_short = generate_ppg_from_ecg(short_signal)
        self.assertEqual(len(ppg_short), len(short_signal))
        
        # Single value signal
        single_signal = np.array([1.0])
        ppg_single = generate_ppg_from_ecg(single_signal)
        self.assertEqual(len(ppg_single), 1)


class TestLSTMModel(unittest.TestCase):
    """Test cases for the LSTM model architecture and functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.seq_len = 100
        self.input_size = 1
        self.device = torch.device('cpu')  # Force CPU for testing
    
    @unittest.skipUnless(LSTM_MODEL_AVAILABLE, "LSTM model not available")
    def test_model_initialization_default_params(self):
        """Test model initialization with default parameters."""
        model = ECGtoPPG_LSTM()
        
        self.assertIsInstance(model, ECGtoPPG_LSTM)
        self.assertEqual(model.input_size, 1)
        self.assertEqual(model.hidden_size, 128)
        self.assertEqual(model.num_layers, 2)
        self.assertTrue(model.bidirectional)
    
    @unittest.skipUnless(LSTM_MODEL_AVAILABLE, "LSTM model not available")
    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        model = ECGtoPPG_LSTM(
            input_size=2,
            hidden_size=64,
            num_layers=3,
            bidirectional=False,
            dropout_rate=0.5
        )
        
        self.assertEqual(model.input_size, 2)
        self.assertEqual(model.hidden_size, 64)
        self.assertEqual(model.num_layers, 3)
        self.assertFalse(model.bidirectional)
        self.assertEqual(model.dropout_rate, 0.5)
    
    @unittest.skipUnless(LSTM_MODEL_AVAILABLE, "LSTM model not available")
    def test_model_forward_pass(self):
        """Test model forward pass with different input shapes."""
        model = ECGtoPPG_LSTM()
        model.eval()
        
        # Test with standard input
        x = torch.randn(self.batch_size, self.seq_len, self.input_size)
        with torch.no_grad():
            output = model(x)
        
        expected_shape = (self.batch_size, self.seq_len, 1)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.isfinite(output).all(), "Model output contains non-finite values")
    
    @unittest.skipUnless(LSTM_MODEL_AVAILABLE, "LSTM model not available")
    def test_model_forward_pass_different_sizes(self):
        """Test model forward pass with different batch and sequence sizes."""
        model = ECGtoPPG_LSTM()
        model.eval()
        
        test_cases = [
            (1, 50, 1),    # Small batch and sequence
            (16, 200, 1),  # Different sizes
            (64, 25, 1),   # Large batch, small sequence
        ]
        
        for batch, seq, input_dim in test_cases:
            with self.subTest(batch=batch, seq=seq, input_dim=input_dim):
                x = torch.randn(batch, seq, input_dim)
                with torch.no_grad():
                    output = model(x)
                self.assertEqual(output.shape, (batch, seq, 1))
    
    @unittest.skipUnless(LSTM_MODEL_AVAILABLE, "LSTM model not available")
    def test_model_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = ECGtoPPG_LSTM()
        x = torch.randn(8, 50, 1, requires_grad=True)
        target = torch.randn(8, 50, 1)
        
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check that gradients exist for model parameters
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for parameter {name}")


class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing and sequence creation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ecg_data = np.sin(np.linspace(0, 10*np.pi, 500))
        self.ppg_data = np.cos(np.linspace(0, 10*np.pi, 500))
    
    @unittest.skipUnless(TESTING_MODULE_AVAILABLE, "Testing module not available")
    def test_create_sequences_basic(self):
        """Test basic sequence creation functionality."""
        seq_len = 50
        X, Y = create_sequences(self.ecg_data, self.ppg_data, seq_len=seq_len)
        
        expected_num_sequences = len(self.ecg_data) - seq_len
        self.assertEqual(X.shape, (expected_num_sequences, seq_len))
        self.assertEqual(Y.shape, (expected_num_sequences, seq_len))
    
    @unittest.skipUnless(TESTING_MODULE_AVAILABLE, "Testing module not available")
    def test_create_sequences_edge_cases(self):
        """Test sequence creation with edge cases."""
        # Sequence length equal to data length
        seq_len = len(self.ecg_data)
        X, Y = create_sequences(self.ecg_data, self.ppg_data, seq_len=seq_len)
        self.assertEqual(len(X), 0)
        self.assertEqual(len(Y), 0)
        
        # Very short data
        short_ecg = np.array([1, 2, 3])
        short_ppg = np.array([4, 5, 6])
        X, Y = create_sequences(short_ecg, short_ppg, seq_len=2)
        self.assertEqual(X.shape, (1, 2))
        self.assertEqual(Y.shape, (1, 2))
    
    @unittest.skipUnless(TESTING_MODULE_AVAILABLE, "Testing module not available")
    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        y_true = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y_pred = np.array([[1.1, 1.9, 3.1], [3.9, 5.1, 5.9]])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        required_metrics = ['mse', 'rmse', 'mae', 'r2', 'pearson']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertTrue(np.isfinite(metrics[metric]))
    
    @unittest.skipUnless(TESTING_MODULE_AVAILABLE, "Testing module not available")
    def test_calculate_metrics_perfect_prediction(self):
        """Test metrics with perfect prediction."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        
        metrics = calculate_metrics(y_true, y_pred)
        
        self.assertAlmostEqual(metrics['mse'], 0.0, places=10)
        self.assertAlmostEqual(metrics['rmse'], 0.0, places=10)
        self.assertAlmostEqual(metrics['mae'], 0.0, places=10)
        self.assertAlmostEqual(metrics['r2'], 1.0, places=10)
        self.assertAlmostEqual(metrics['pearson'], 1.0, places=10)


class TestFileOperations(unittest.TestCase):
    """Test cases for file operations and data loading."""
    
    def test_create_temporary_csv_and_load(self):
        """Test creating and loading a temporary CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write sample ECG data
            f.write("ECG Amplitude\n")
            for i in range(100):
                f.write(f"{np.sin(i * 0.1):.6f}\n")
            temp_file = f.name
        
        try:
            # Test loading the data
            data = pd.read_csv(temp_file)
            ecg = data["ECG Amplitude"].values
            
            self.assertEqual(len(ecg), 100)
            self.assertIsInstance(ecg, np.ndarray)
            self.assertTrue(np.isfinite(ecg).all())
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    def test_model_save_load_simulation(self):
        """Test model saving and loading simulation."""
        if not LSTM_MODEL_AVAILABLE:
            self.skipTest("LSTM model not available")
        
        model = ECGtoPPG_LSTM()
        
        # Set model to eval mode for consistent results
        model.eval()
        
        # Create temporary file for model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_model_file = f.name
        
        try:
            # Save model
            torch.save(model.state_dict(), temp_model_file)
            
            # Load model
            loaded_model = ECGtoPPG_LSTM()
            loaded_model.load_state_dict(torch.load(temp_model_file, weights_only=True))
            loaded_model.eval()
            
            # Test that loaded model produces same output
            # Use a fixed seed for reproducible results
            torch.manual_seed(42)
            x = torch.randn(1, 50, 1)
            
            with torch.no_grad():
                torch.manual_seed(42)  # Reset seed for consistent dropout behavior
                original_output = model(x)
                torch.manual_seed(42)  # Reset seed for loaded model
                loaded_output = loaded_model(x)
            
            # Check that the models have the same parameters
            for (name1, param1), (name2, param2) in zip(model.named_parameters(), loaded_model.named_parameters()):
                self.assertEqual(name1, name2)
                self.assertTrue(torch.allclose(param1, param2, atol=1e-6))
            
            # The outputs should be close but may not be identical due to batch norm running stats
            self.assertTrue(torch.allclose(original_output, loaded_output, atol=1e-3))
            
        finally:
            os.unlink(temp_model_file)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    @unittest.skipUnless(all([LSTM_MODEL_AVAILABLE, TRAINING_MODULE_AVAILABLE, TESTING_MODULE_AVAILABLE]), 
                         "Required modules not available")
    def test_end_to_end_pipeline(self):
        """Test the complete pipeline from ECG to PPG prediction."""
        # Generate synthetic ECG data
        ecg_data = np.sin(np.linspace(0, 20*np.pi, 2000)) + 0.1*np.random.randn(2000)
        
        # Generate PPG from ECG
        ppg_data = generate_ppg_from_ecg(ecg_data)
        
        # Normalize signals
        ecg_normalized = (ecg_data - ecg_data.mean()) / ecg_data.std()
        ppg_normalized = (ppg_data - ppg_data.mean()) / ppg_data.std()
        
        # Create sequences
        X, Y = create_sequences(ecg_normalized, ppg_normalized, seq_len=100)
        
        # Prepare data for model
        X = X[:, :, np.newaxis]
        Y = Y[:, :, np.newaxis]
        
        # Initialize and test model
        model = ECGtoPPG_LSTM()
        model.eval()
        
        # Make prediction on a small batch
        test_batch = torch.tensor(X[:5]).float()
        with torch.no_grad():
            predictions = model(test_batch)
        
        # Verify prediction shape and properties
        self.assertEqual(predictions.shape, (5, 100, 1))
        self.assertTrue(torch.isfinite(predictions).all())
        
        # Calculate metrics
        test_labels = Y[:5]
        metrics = calculate_metrics(test_labels, predictions.numpy())
        
        # Verify all metrics are calculated
        required_metrics = ['mse', 'rmse', 'mae', 'r2', 'pearson']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertTrue(np.isfinite(metrics[metric]))


def run_tests():
    """Run all tests and provide a summary."""
    print("=" * 60)
    print("PPG ESTIMATION PROJECT - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print()
    
    # Check module availability
    print("Module Availability Check:")
    print(f"  LSTM Model: {'✓' if LSTM_MODEL_AVAILABLE else '✗'}")
    print(f"  Training Module: {'✓' if TRAINING_MODULE_AVAILABLE else '✗'}")
    print(f"  Testing Module: {'✓' if TESTING_MODULE_AVAILABLE else '✗'}")
    print()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDataGeneration,
        TestLSTMModel, 
        TestDataProcessing,
        TestFileOperations,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Check if running in verbose mode
    if len(sys.argv) > 1 and sys.argv[1] in ['-v', '--verbose']:
        run_tests()
    else:
        # Run with standard unittest
        unittest.main()
