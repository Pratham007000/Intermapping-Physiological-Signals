/*
Arduino Nano 33 BLE Sense - TinyML ECG-to-PPG Conversion
========================================================

Real-time ECG-to-PPG conversion using TinyML LSTM model.
Demonstrates complete pipeline including:
- ECG signal acquisition from analog input
- Real-time LSTM inference 
- PPG signal generation and output
- Heart rate estimation
- Bluetooth data transmission
- Performance monitoring

Hardware Connections:
- ECG Input: A0 (analog input with proper ECG frontend)
- PPG Output: D3 (PWM output for PPG visualization)
- LED Status: D13 (built-in LED for heartbeat indication)
- Serial: USB (115200 baud for data logging)

LIBRARY DEPENDENCIES FOR ARDUINO IDE:
=====================================
To use this code, install the following libraries through Arduino IDE Library Manager:
1. ArduinoBLE (by Arduino)
2. Arduino_LSM9DS1 (for IMU - optional)
3. PDM (for microphone - optional)

BOARD SELECTION:
================
Select: Arduino Nano 33 BLE
NOTE: This code is specifically designed for ARM Cortex-M4 processor

Author: Arduino TinyML PPG Team
Version: 2.0
Date: 2025
*/

// Board compatibility check
#if !defined(ARDUINO_ARCH_MBED_NANO) && !defined(ARDUINO_ARCH_MBED)
  #error "This code is designed specifically for Arduino Nano 33 BLE Sense. Please select the correct board in Tools > Board."
#endif

#include "arduino_nano_tinyml_model.h"
#include <ArduinoBLE.h>

// Optional sensor includes (comment out if not using)
// #include <Arduino_LSM9DS1.h>  // For IMU data
// #include <PDM.h>              // For microphone data

// Pin definitions
#define ECG_INPUT_PIN A0
#define PPG_OUTPUT_PIN 3
#define HEARTBEAT_LED_PIN LED_BUILTIN
#define STATUS_LED_PIN 2

// Timing constants
#define SAMPLING_RATE 360          // Hz - ECG sampling rate
#define SAMPLING_INTERVAL_US (1000000 / SAMPLING_RATE)
#define HEARTBEAT_LED_DURATION 100 // ms
#define STATUS_UPDATE_INTERVAL 1000 // ms
#define CALIBRATION_SAMPLES 1000   // Number of samples for initial calibration

// BLE Service and Characteristics
BLEService ppgService("12345678-1234-1234-1234-123456789abc");
BLECharacteristic ecgCharacteristic("12345678-1234-1234-1234-123456789abd", BLERead | BLENotify, 20);
BLECharacteristic ppgCharacteristic("12345678-1234-1234-1234-123456789abe", BLERead | BLENotify, 20);
BLECharacteristic heartRateCharacteristic("12345678-1234-1234-1234-123456789abf", BLERead | BLENotify, 4);

// Global objects
ArduinoTinyMLModel tinyml_model;

// Global variables
unsigned long last_sample_time = 0;
unsigned long last_heartbeat_time = 0;
unsigned long last_status_update = 0;
float calibration_buffer[CALIBRATION_SAMPLES];
bool model_initialized = false;
bool calibration_complete = false;
uint16_t calibration_index = 0;

// Performance tracking
uint32_t total_samples_processed = 0;
float average_ecg_value = 0.0;
float average_ppg_value = 0.0;

void setup() {
    Serial.begin(SERIAL_BAUD_RATE);
    
    // Wait for serial connection in debug mode
    #if ENABLE_SERIAL_DEBUG
    while (!Serial && millis() < 5000) {
        delay(10);
    }
    #endif
    
    Serial.println("Arduino Nano 33 BLE Sense - TinyML ECG-to-PPG v2.0");
    Serial.println("====================================================");
    Serial.println("Board: Arduino Nano 33 BLE Sense (ARM Cortex-M4)");
    Serial.print("CPU Frequency: ");
    Serial.print(SystemCoreClock / 1000000);
    Serial.println(" MHz");
    
    // Initialize pins
    pinMode(ECG_INPUT_PIN, INPUT);
    pinMode(PPG_OUTPUT_PIN, OUTPUT);
    pinMode(HEARTBEAT_LED_PIN, OUTPUT);
    pinMode(STATUS_LED_PIN, OUTPUT);
    
    // Initialize BLE
    if (!BLE.begin()) {
        Serial.println("Failed to initialize BLE!");
        while (1) {
            digitalWrite(STATUS_LED_PIN, HIGH);
            delay(100);
            digitalWrite(STATUS_LED_PIN, LOW);
            delay(100);
        }
    }
    
    // Setup BLE service and characteristics
    setup_ble_service();
    
    Serial.print("Free memory: ");
    Serial.print(get_free_memory());
    Serial.println(" bytes");
    
    // Initialize TinyML model
    Serial.println("Initializing TinyML model...");
    if (tinyml_model.initialize()) {
        model_initialized = true;
        Serial.println("Model initialized successfully!");
        tinyml_model.print_model_info();
    } else {
        Serial.println("Failed to initialize model!");
        while (1) {
            digitalWrite(STATUS_LED_PIN, HIGH);
            delay(500);
            digitalWrite(STATUS_LED_PIN, LOW);
            delay(500);
        }
    }
    
    // Run self-test
    if (tinyml_model.self_test()) {
        Serial.println("Self-test passed!");
    } else {
        Serial.println("Warning: Self-test failed!");
    }
    
    // Start calibration phase
    Serial.println("Starting calibration phase...");
    Serial.print("Collecting ");
    Serial.print(CALIBRATION_SAMPLES);
    Serial.println(" samples for calibration...");
    
    digitalWrite(STATUS_LED_PIN, HIGH); // Status LED on during calibration
    
    Serial.println("Setup complete. Starting real-time processing...");
    Serial.println("Format: timestamp,ecg,ppg,heart_rate");
}

void loop() {
    unsigned long current_time = micros();
    
    // Check if it's time for next sample
    if (current_time - last_sample_time >= SAMPLING_INTERVAL_US) {
        last_sample_time = current_time;
        
        // Read ECG sample
        float ecg_sample = tinyml_model.read_ecg_from_adc(ECG_INPUT_PIN);
        
        // Handle calibration phase
        if (!calibration_complete) {
            handle_calibration(ecg_sample);
            return;
        }
        
        // Process sample through TinyML model
        float ppg_sample = tinyml_model.predict_ppg_sample(ecg_sample);
        
        // Output PPG signal
        tinyml_model.output_ppg_to_pwm(ppg_sample, PPG_OUTPUT_PIN);
        
        // Update running averages
        update_running_averages(ecg_sample, ppg_sample);
        
        // Handle heartbeat LED
        handle_heartbeat_led();
        
        // Send data via BLE
        send_ble_data(ecg_sample, ppg_sample);
        
        // Log data to serial
        tinyml_model.log_data_to_serial(ecg_sample, ppg_sample);
        
        total_samples_processed++;
    }
    
    // Handle BLE connections
    handle_ble_connections();
    
    // Periodic status updates
    handle_status_updates();
    
    // Handle serial commands
    handle_serial_commands();
}

void setup_ble_service() {
    BLE.setLocalName("TinyML-PPG");
    BLE.setAdvertisedService(ppgService);
    
    ppgService.addCharacteristic(ecgCharacteristic);
    ppgService.addCharacteristic(ppgCharacteristic);
    ppgService.addCharacteristic(heartRateCharacteristic);
    
    BLE.addService(ppgService);
    
    // Set initial values (cast float to int32_t for BLE)
    ecgCharacteristic.writeValue((int32_t)0);
    ppgCharacteristic.writeValue((int32_t)0);
    heartRateCharacteristic.writeValue((int32_t)60);
    
    BLE.advertise();
    Serial.println("BLE service started. Advertising as 'TinyML-PPG'");
}

void handle_calibration(float ecg_sample) {
    if (calibration_index < CALIBRATION_SAMPLES) {
        calibration_buffer[calibration_index] = ecg_sample;
        calibration_index++;
        
        // Show calibration progress
        if (calibration_index % 100 == 0) {
            Serial.print("Calibration progress: ");
            Serial.print((calibration_index * 100) / CALIBRATION_SAMPLES);
            Serial.println("%");
        }
    } else {
        // Complete calibration
        Serial.println("Calibration complete. Processing calibration data...");
        
        tinyml_model.calibrate_input(calibration_buffer, CALIBRATION_SAMPLES);
        calibration_complete = true;
        
        digitalWrite(STATUS_LED_PIN, LOW); // Turn off status LED
        
        Serial.println("Model calibrated and ready for real-time processing!");
    }
}

void update_running_averages(float ecg_sample, float ppg_sample) {
    // Simple exponential moving average
    const float alpha = 0.01; // Smoothing factor
    
    average_ecg_value = alpha * ecg_sample + (1.0 - alpha) * average_ecg_value;
    average_ppg_value = alpha * ppg_sample + (1.0 - alpha) * average_ppg_value;
}

void handle_heartbeat_led() {
    // Get current heart rate estimate
    float heart_rate = tinyml_model.get_heart_rate_estimate();
    
    // Calculate expected interval between heartbeats
    uint32_t heartbeat_interval_ms = (uint32_t)(60000.0 / heart_rate);
    
    // Check if it's time for next heartbeat LED flash
    unsigned long current_time = millis();
    if (current_time - last_heartbeat_time >= heartbeat_interval_ms) {
        digitalWrite(HEARTBEAT_LED_PIN, HIGH);
        last_heartbeat_time = current_time;
        
        // Schedule LED turn-off
        // Note: This is a simple approach; for better accuracy, use a timer
    }
    
    // Turn off LED after duration
    if (digitalRead(HEARTBEAT_LED_PIN) == HIGH && 
        (current_time - last_heartbeat_time >= HEARTBEAT_LED_DURATION)) {
        digitalWrite(HEARTBEAT_LED_PIN, LOW);
    }
}

void send_ble_data(float ecg_sample, float ppg_sample) {
    if (BLE.connected()) {
        // Send ECG value (convert float to int32_t for BLE)
        int32_t ecg_int = (int32_t)(ecg_sample * 1000); // Scale by 1000 for precision
        ecgCharacteristic.writeValue(ecg_int);
        
        // Send PPG value (convert float to int32_t for BLE)
        int32_t ppg_int = (int32_t)(ppg_sample * 1000); // Scale by 1000 for precision
        ppgCharacteristic.writeValue(ppg_int);
        
        // Send heart rate (less frequently to avoid overwhelming)
        static uint32_t ble_counter = 0;
        if (ble_counter % 10 == 0) { // Send heart rate every 10 samples
            float heart_rate = tinyml_model.get_heart_rate_estimate();
            int32_t hr_int = (int32_t)heart_rate; // Heart rate as integer BPM
            heartRateCharacteristic.writeValue(hr_int);
        }
        ble_counter++;
    }
}

void handle_ble_connections() {
    BLEDevice central = BLE.central();
    
    static bool was_connected = false;
    bool is_connected = central.connected();
    
    // Connection status changes
    if (is_connected && !was_connected) {
        Serial.print("Connected to central: ");
        Serial.println(central.address());
        digitalWrite(STATUS_LED_PIN, HIGH);
    } else if (!is_connected && was_connected) {
        Serial.println("Disconnected from central");
        digitalWrite(STATUS_LED_PIN, LOW);
    }
    
    was_connected = is_connected;
}

void handle_status_updates() {
    unsigned long current_time = millis();
    
    if (current_time - last_status_update >= STATUS_UPDATE_INTERVAL) {
        last_status_update = current_time;
        
        if (calibration_complete) {
            print_status_summary();
        }
    }
}

void print_status_summary() {
    Serial.println("=== Status Summary ===");
    Serial.print("Uptime: ");
    Serial.print(millis() / 1000);
    Serial.println(" seconds");
    
    Serial.print("Samples processed: ");
    Serial.println(total_samples_processed);
    
    Serial.print("Sample rate: ");
    if (total_samples_processed > 0) {
        float actual_rate = (float)total_samples_processed * 1000.0 / millis();
        Serial.print(actual_rate, 1);
        Serial.println(" Hz");
    } else {
        Serial.println("N/A");
    }
    
    Serial.print("Average ECG: ");
    Serial.println(average_ecg_value, 4);
    
    Serial.print("Average PPG: ");
    Serial.println(average_ppg_value, 4);
    
    Serial.print("Heart rate: ");
    Serial.print(tinyml_model.get_heart_rate_estimate(), 1);
    Serial.println(" BPM");
    
    Serial.print("Peak count: ");
    Serial.println(tinyml_model.get_peak_count());
    
    Serial.print("BLE status: ");
    Serial.println(BLE.connected() ? "Connected" : "Disconnected");
    
    Serial.print("Free memory: ");
    Serial.print(get_free_memory());
    Serial.println(" bytes");
    
    // Print model performance metrics
    tinyml_model.print_performance_metrics();
    
    Serial.println("=====================");
}

void handle_serial_commands() {
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');
        command.trim();
        
        if (command == "status") {
            print_status_summary();
        } else if (command == "reset") {
            Serial.println("Resetting model...");
            tinyml_model.reset_model();
            Serial.println("Model reset complete.");
        } else if (command == "calibrate") {
            if (!calibration_complete) {
                Serial.println("Calibration already in progress.");
            } else {
                Serial.println("Restarting calibration...");
                calibration_complete = false;
                calibration_index = 0;
                digitalWrite(STATUS_LED_PIN, HIGH);
            }
        } else if (command == "test") {
            Serial.println("Running self-test...");
            bool result = tinyml_model.self_test();
            Serial.print("Self-test result: ");
            Serial.println(result ? "PASSED" : "FAILED");
        } else if (command == "help") {
            print_help();
        } else if (command.length() > 0) {
            Serial.print("Unknown command: ");
            Serial.println(command);
            Serial.println("Type 'help' for available commands.");
        }
    }
}

void print_help() {
    Serial.println("=== Available Commands ===");
    Serial.println("status    - Print current status and metrics");
    Serial.println("reset     - Reset the TinyML model state");
    Serial.println("calibrate - Restart input calibration");
    Serial.println("test      - Run model self-test");
    Serial.println("help      - Show this help message");
    Serial.println("==========================");
}

// Error handling function
void handle_error(const char* error_message) {
    Serial.print("ERROR: ");
    Serial.println(error_message);
    
    // Flash status LED rapidly to indicate error
    for (int i = 0; i < 10; i++) {
        digitalWrite(STATUS_LED_PIN, HIGH);
        delay(100);
        digitalWrite(STATUS_LED_PIN, LOW);
        delay(100);
    }
}

// Memory monitoring function
void check_memory_usage() {
    uint32_t free_memory = get_free_memory();
    
    if (free_memory < 1000) { // Less than 1KB free
        Serial.print("WARNING: Low memory! Free: ");
        Serial.print(free_memory);
        Serial.println(" bytes");
        
        // Could implement memory cleanup here
    }
}
