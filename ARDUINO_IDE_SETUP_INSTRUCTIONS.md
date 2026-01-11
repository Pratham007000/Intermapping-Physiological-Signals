# 🎯 Arduino IDE Setup Instructions - UPDATED

## ⚠️ IMPORTANT: Proper Folder Structure Required

Arduino IDE requires all sketch files to be in a folder with the **same name as the main .ino file**.

## 📁 Ready-to-Use Folders Created

I've created the proper folder structure for you:

```
PPG_Estimation_Project/
├── arduino_nano_main/                    # 👈 FULL TinyML VERSION
│   ├── arduino_nano_main.ino            # Main sketch
│   ├── arduino_nano_tinyml_model.h      # Header file  
│   ├── arduino_nano_tinyml_model.cpp    # Implementation
│   └── arduino_nano_model_weights.cpp   # Model weights
│
├── arduino_nano_simple_demo/             # 👈 SIMPLE DEMO VERSION
│   └── arduino_nano_simple_demo.ino     # Simple demo (recommended for testing)
│
└── README files...
```

## 🚀 Quick Start Guide

### Step 1: Arduino IDE Setup
1. **Install Arduino IDE 2.x** from [arduino.cc](https://www.arduino.cc/en/software)
2. Open Arduino IDE

### Step 2: Install Board Support
1. Go to **Tools > Board > Boards Manager**
2. Search for **"Arduino Mbed OS Nano Boards"**
3. Install the package
4. Select **Tools > Board > Arduino Mbed OS Nano Boards > Arduino Nano 33 BLE**

### Step 3: Install Required Library
1. Go to **Tools > Manage Libraries**
2. Search for **"ArduinoBLE"**
3. Install **ArduinoBLE by Arduino**

### Step 4: Open the Correct Sketch

#### Option A: Simple Demo (RECOMMENDED for first test)
1. In Arduino IDE: **File > Open**
2. Navigate to: `/Users/prathamarunshetty/Desktop/PPG_Estimation_Project/arduino_nano_simple_demo/`
3. Open: **`arduino_nano_simple_demo.ino`**
4. This will automatically load all required files

#### Option B: Full TinyML Version 
1. In Arduino IDE: **File > Open**
2. Navigate to: `/Users/prathamarunshetty/Desktop/PPG_Estimation_Project/arduino_nano_main/`
3. Open: **`arduino_nano_main.ino`**
4. This will automatically load all 4 files (main + header + implementation + weights)

### Step 5: Verify Settings
Check these settings in Arduino IDE:
- **Board**: Arduino Nano 33 BLE ✅
- **Port**: Select your board's port
- **Programmer**: Default (don't change)

### Step 6: Compile and Upload
1. Click **Verify** (✓ checkmark) to compile
2. Connect your Arduino Nano 33 BLE Sense
3. Select the correct **Port** under Tools > Port
4. Click **Upload** (→ arrow)

## 🔧 Troubleshooting

### Error: "No such file or directory"
**Problem**: Files aren't in the same folder
**Solution**: Use the folder structure I created above. Don't copy individual files.

### Error: "Board not selected correctly"
**Problem**: Wrong board selected
**Solution**: Select **Arduino Nano 33 BLE** (not Nano 33 IoT)

### Error: "ArduinoBLE.h not found"
**Problem**: Library not installed
**Solution**: Install ArduinoBLE via Library Manager

### Error: "Board doesn't support this architecture"
**Problem**: Using wrong board
**Solution**: Make sure you have the real Arduino Nano 33 BLE Sense, not a different board

## 📋 Hardware Requirements

- **Arduino Nano 33 BLE Sense** (not Nano 33 IoT!)
- **ECG sensor** connected to pin A0 (or potentiometer for testing)
- **USB cable** for programming

## 🎮 Testing

### Simple Demo Test:
1. Open Serial Monitor (115200 baud)
2. Connect a potentiometer to A0 
3. Turn the potentiometer - you should see ECG/PPG values change
4. Built-in LED will flash like a heartbeat

### Full TinyML Test:
1. Same as above but with real LSTM model running
2. Will show calibration process first
3. Type commands like `status`, `help` in Serial Monitor

## 📱 BLE Connectivity

Both versions support Bluetooth:
- **Device name**: "TinyML-PPG" (full) or "TinyML-Demo" (simple)
- Use any BLE scanner app to connect and see data

## 🎯 Which Version to Use?

**Start with Simple Demo** if:
- First time using this code
- Want to test hardware connections
- Need basic ECG-to-PPG conversion

**Use Full TinyML Version** if:
- Hardware is working with simple demo
- Want the real LSTM model
- Need advanced features like calibration

## ✅ Success Indicators

You'll know it's working when you see:

```
Arduino Nano 33 BLE Sense - TinyML ECG-to-PPG v2.0
====================================================
Board: Arduino Nano 33 BLE Sense (ARM Cortex-M4)
CPU Frequency: 64 MHz
BLE initialized. Device name: TinyML-PPG
Model initialized successfully!
Setup complete. Starting real-time processing...
```

## 📞 Still Having Issues?

1. **Double-check board selection**: Must be "Arduino Nano 33 BLE"
2. **Verify folder structure**: Use the folders I created
3. **Check library installation**: ArduinoBLE must be installed
4. **Try simple demo first**: Start with the simpler version

The folders are ready to go - just open the .ino file from the correct folder in Arduino IDE!
