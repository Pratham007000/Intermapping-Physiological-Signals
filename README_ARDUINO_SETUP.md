# Arduino Nano 33 BLE Sense TinyML ECG-to-PPG Setup Guide

## Overview

This project implements a TinyML LSTM model for real-time ECG-to-PPG conversion on the Arduino Nano 33 BLE Sense. The model runs entirely on the microcontroller and provides real-time heart rate monitoring with Bluetooth connectivity.

## Hardware Requirements

- **Arduino Nano 33 BLE Sense** (ARM Cortex-M4 @ 64 MHz, 256KB SRAM, 1MB Flash)
- **ECG sensor circuit** connected to analog pin A0
- **LEDs** for status indication (built-in LED used for heartbeat)
- **USB cable** for programming and serial communication

## Arduino IDE Setup

### Step 1: Install Arduino IDE

1. Download and install Arduino IDE 2.x from [arduino.cc](https://www.arduino.cc/en/software)
2. Launch Arduino IDE

### Step 2: Install Board Support

1. Go to **Tools > Board > Boards Manager**
2. Search for "Arduino Mbed OS Nano Boards"
3. Install the package (this includes support for Nano 33 BLE)
4. Select **Tools > Board > Arduino Mbed OS Nano Boards > Arduino Nano 33 BLE**

### Step 3: Install Required Libraries

Install the following libraries via **Tools > Manage Libraries**:

1. **ArduinoBLE** (by Arduino) - Required for Bluetooth functionality
2. **Arduino_LSM9DS1** (by Arduino) - Optional, for IMU data
3. **PDM** (by Arduino) - Optional, for microphone data

### Step 4: Board Configuration

Select the following settings in the Arduino IDE:

- **Board**: Arduino Nano 33 BLE
- **Port**: Select your board's COM port
- **Programmer**: Use default

## Project Files Structure

```
PPG_Estimation_Project/
├── arduino_nano_main.ino              # Main sketch file
├── arduino_nano_tinyml_model.h        # Model header file
├── arduino_nano_tinyml_model.cpp      # Model implementation
├── arduino_nano_model_weights.cpp     # Pre-trained model weights
└── README_ARDUINO_SETUP.md            # This setup guide
```

## Installation Steps

### Step 1: Download Project Files

Ensure all four files are in the same folder:
- `arduino_nano_main.ino`
- `arduino_nano_tinyml_model.h`
- `arduino_nano_tinyml_model.cpp`
- `arduino_nano_model_weights.cpp`

### Step 2: Open Main Sketch

1. Double-click `arduino_nano_main.ino` to open it in Arduino IDE
2. The IDE should automatically detect and open all related files (.h, .cpp)

### Step 3: Verify Board Selection

Check that the board is correctly selected by looking for this in the code:
```cpp
// Board compatibility check
#if !defined(ARDUINO_ARCH_MBED_NANO) && !defined(ARDUINO_ARCH_MBED)
  #error "This code is designed specifically for Arduino Nano 33 BLE Sense. Please select the correct board in Tools > Board."
#endif
```

### Step 4: Compile and Upload

1. Click **Verify** (checkmark icon) to compile the code
2. Connect your Arduino Nano 33 BLE Sense via USB
3. Select the correct port under **Tools > Port**
4. Click **Upload** (arrow icon) to flash the code

## Hardware Connections

### ECG Input
- Connect your ECG sensor output to **A0** (analog input)
- Ensure proper signal conditioning (amplification, filtering)
- Signal range should be within 0-3.3V

### Output Connections
- **Built-in LED (D13)**: Heartbeat indicator
- **Pin D2**: Status LED (optional)
- **Pin D3**: PPG output via PWM (optional, for visualization)

### Power
- USB power is sufficient for development
- For battery operation, use the VIN pin with 5-21V input

## Usage

### Serial Monitor Setup

1. Open **Tools > Serial Monitor**
2. Set baud rate to **115200**
3. You should see initialization messages

### Expected Serial Output

```
Arduino Nano 33 BLE Sense - TinyML ECG-to-PPG v2.0
====================================================
Board: Arduino Nano 33 BLE Sense (ARM Cortex-M4)
CPU Frequency: 64 MHz
Free memory: 200000 bytes
Initializing TinyML model...
Model initialized successfully!
=== Model Information ===
Input size: 1
Hidden size: 16
Number of layers: 1
Sequence length: 16
Memory usage: 2048 bytes
Self-test passed!
Starting calibration phase...
Collecting 1000 samples for calibration...
```

### Serial Commands

Type these commands in the Serial Monitor:

- `status` - Print current status and metrics
- `reset` - Reset the TinyML model state
- `calibrate` - Restart input calibration
- `test` - Run model self-test
- `help` - Show available commands

### Bluetooth Connectivity

The device advertises as **"TinyML-PPG"** and provides:
- ECG data characteristic
- PPG data characteristic  
- Heart rate characteristic

## Calibration Process

1. The system automatically collects 1000 ECG samples for calibration
2. During calibration, the status LED remains on
3. Progress is shown in the serial monitor
4. After calibration, real-time processing begins

## Performance Characteristics

- **Model Size**: ~8KB
- **RAM Usage**: ~2KB
- **Inference Time**: ~100-500 µs per sample
- **Max Sample Rate**: ~2000 Hz (theoretical)
- **Actual Sample Rate**: 360 Hz (configured)
- **Power Consumption**: ~20mA (active)

## Troubleshooting

### Compilation Errors

**Board Selection Error**:
```
This code is designed specifically for Arduino Nano 33 BLE Sense
```
- **Solution**: Select **Arduino Nano 33 BLE** in Tools > Board

**Library Missing**:
```
fatal error: ArduinoBLE.h: No such file or directory
```
- **Solution**: Install ArduinoBLE library via Library Manager

### Runtime Issues

**BLE Initialization Failed**:
- Check that the board is an actual Nano 33 BLE (not Nano 33 IoT)
- Try resetting the board

**Memory Issues**:
- The code is optimized for 256KB SRAM
- Reduce buffer sizes if needed
- Monitor free memory via serial output

**ECG Signal Issues**:
- Ensure proper ECG sensor connection
- Check signal amplification and filtering
- Verify 0-3.3V signal range

### Performance Tuning

**Adjust Sample Rate**:
```cpp
#define SAMPLING_RATE 360  // Change this value
```

**Model Parameters**:
```cpp
#define HIDDEN_SIZE 16      // Reduce for less memory usage
#define SEQUENCE_LENGTH 16  // Reduce for faster inference
```

## Advanced Features

### Optional Sensor Integration

Uncomment these includes to add sensor data:
```cpp
#include <Arduino_LSM9DS1.h>  // For IMU data
#include <PDM.h>              // For microphone data
```

### Custom ECG Processing

Modify the `read_ecg_from_adc()` function for your specific ECG frontend:
```cpp
float ArduinoTinyMLModel::read_ecg_from_adc(uint8_t pin) {
    // Add your custom ECG processing here
}
```

### Model Retraining

To use your own trained model:
1. Run the Python training scripts
2. Export weights in the correct format
3. Replace values in `arduino_nano_model_weights.cpp`

## Support and Development

- Ensure Arduino IDE 2.x is used for best compatibility
- Monitor serial output for debugging information
- Use the built-in LED for visual heartbeat confirmation
- BLE characteristics can be monitored with smartphone apps

## Model Architecture

- **Input**: Single ECG value
- **LSTM**: 16 hidden units, 1 layer
- **Output**: Single PPG value
- **Fixed-point arithmetic** for efficiency
- **Sequence length**: 16 samples

This implementation provides a complete TinyML solution for real-time ECG-to-PPG conversion suitable for wearable health monitoring applications.
