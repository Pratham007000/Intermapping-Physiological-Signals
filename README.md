# Intermapping of Physiological Signals 

**Research Internship Project | IIIT Bangalore** *May 2025 - Aug 2025*

![Status](https://img.shields.io/badge/Status-Research_Complete-success)
![Platform](https://img.shields.io/badge/Platform-Arduino_Nano%20%7C%20RPi_Pico-blue)
![Models](https://img.shields.io/badge/AI-Transformers%20%7C%20LSTM%20%7C%20TinyML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview
This project explores the **interrelationships between physiological signals**, leveraging Deep Learning to map and synthesize signals across different modalities. The core innovation is the ability to estimate complex hemodynamic parameters (like ECG, SCG, and Blood Pressure) using only a standard PPG input.

The system features a complete **Edge AI pipeline**, allowing optimized models to run on resource-constrained microcontrollers (Arduino Nano 33 BLE Sense and Raspberry Pi Pico) for real-time, wearable monitoring.

## 🚀 Key Features

### 1. Cross-Modal Signal Synthesis (Intermapping)
Using advanced architectures (Hybrid Transformer-CNNs, Bi-LSTMs), we map:
* **PPG ➡️ ECG:** Synthesizing optical blood flow waveforms from electrical heart activity.
* **ECG ➡️ SCG:** Predicting mechanical heart vibrations (Seismocardiogram).
* **ECG ➡️ GSR/EMG:** Estimating Electrodermal and Electromyographic activity.

### 2. Hemodynamic Parameter Estimation
Beyond signal reconstruction, the system estimates clinical biomarkers:
* **Vital Signs:** Heart Rate (HR), Heart Rate Variability (HRV), Respiratory Rate.
* **Advanced Metrics:** Blood Pressure (BP), Cardiac Output (CO), Pulse Wave Velocity (PWV), and SpO2.

### 3. Edge AI & TinyML Deployment
* **Arduino Nano 33 BLE Sense:** C++ implementation using TensorFlow Lite for Microcontrollers.
* **Raspberry Pi Pico:** Custom MicroPython inference engine.
* **Optimization:** Weights are quantized and converted to C++ byte arrays/MicroPython lists for <50KB memory footprint.

### 4. Real-Time Hardware Interface
* **Bluetooth (BLE):** Wireless real-time plotting of inferred signals (`visualize_ppg_ble.py`).
* **HIL Testing:** Hardware-in-the-Loop validation using `arduino_ppg_realtime.py`.

---

## 📂 Repository Organization

### 🧠 Models (`src/models/`)
* `spo2_transformer_cnn.py`: State-of-the-art Hybrid Transformer-CNN for robust SpO2 estimation.
* `lstm_ppg_model.py`: Baseline LSTM architecture for temporal signal mapping.
* `tinyml_lstm_model.py`: Highly optimized, low-parameter model designed for edge deployment.

### 📉 Analysis (`src/analysis/`)
* `hrv_analysis.py` & `prv_analysis.py`: Extracts time/frequency-domain metrics for autonomic nervous system assessment.
* `bp_estimation_mimic.py`: Regression models trained on MIMIC datasets for cuffless BP estimation.
* `respiratory_rate_prediction.py`: Derives breathing rate from signal amplitude modulation (AM) and frequency modulation (FM).

### 🛠️ Firmware (`firmware/`)
* **Arduino:** Contains the `.ino` sketch and `.cpp` model weights. Supports "CSV Playback" mode to test models without a signal generator.
* **Pico:** Pure MicroPython implementation (`pico_main.py`) proving platform independence.

---

## ⚡ Quick Start

### 1. Installation
```bash
git clone [https://github.com/yourusername/Intermapping-Physiological-Signals.git](https://github.com/yourusername/Intermapping-Physiological-Signals.git)
cd Intermapping-Physiological-Signals
pip install -r requirements.txt
