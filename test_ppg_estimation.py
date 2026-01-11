import unittest
import numpy as np
import torch
from train_improved_ppg_model import generate_ppg_from_ecg, load_ecg_data
from lstm_ppg_model import ECGtoPPG_LSTM
from ecg_to_ppg_testing import create_sequences, calculate_metrics

class TestPPGEstimation(unittest.TestCase):
    def test_load_ecg_data(self):
        ecg_data = load_ecg_data()
        self.assertIsInstance(ecg_data, np.ndarray)
        self.assertGreater(len(ecg_data), 0)

    def test_generate_ppg_from_ecg(self):
        ecg_data = np.random.rand(1000)
        ppg_signal = generate_ppg_from_ecg(ecg_data)
        self.assertIsNotNone(ppg_signal)
        self.assertEqual(len(ppg_signal), len(ecg_data))

    def test_lstm_model_initialization(self):
        model = ECGtoPPG_LSTM()
        self.assertIsInstance(model, ECGtoPPG_LSTM)

    def test_create_sequences_functionality(self):
        ecg_data = np.random.rand(150)
        ppg_data = np.random.rand(150)
        X, Y = create_sequences(ecg_data, ppg_data, seq_len=50)
        self.assertEqual(X.shape, (100, 50))
        self.assertEqual(Y.shape, (100, 50))

    def test_calculate_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 2.9])
        metrics = calculate_metrics(y_true, y_pred)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('pearson', metrics)

if __name__ == '__main__':
    unittest.main()

