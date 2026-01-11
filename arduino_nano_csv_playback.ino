/*
Arduino Nano 33 BLE Sense - TinyML with CSV ECG Data Playback
=============================================================

This version uses pre-loaded ECG data from CSV file stored in program memory
instead of reading from analog sensors. Perfect for testing the TinyML model
with real ECG data patterns.

Features:
- Plays back real ECG data from CSV file
- Full TinyML LSTM model processing
- Real-time PPG prediction
- Heart rate estimation
- BLE data streaming
- Serial data logging

Hardware Requirements:
- Arduino Nano 33 BLE Sense
- No ECG sensor needed (uses stored data)

Board Selection:
- Tools > Board > Arduino Mbed OS Nano Boards > Arduino Nano 33 BLE
*/

// Board compatibility check
#if !defined(ARDUINO_ARCH_MBED_NANO) && !defined(ARDUINO_ARCH_MBED)
  #error "This code is designed specifically for Arduino Nano 33 BLE Sense. Please select the correct board in Tools > Board."
#endif

#include "arduino_nano_tinyml_model.h"
#include <ArduinoBLE.h>

// Include ECG data from CSV
#include "ecg_data_array.h"

// Pin definitions  
#define PPG_OUTPUT_PIN 3
#define HEARTBEAT_LED_PIN LED_BUILTIN
#define STATUS_LED_PIN 2

// Timing constants
#define SAMPLING_RATE 360          // Hz - ECG sampling rate
#define SAMPLING_INTERVAL_MS (1000 / SAMPLING_RATE)
#define HEARTBEAT_LED_DURATION 100 // ms
#define STATUS_UPDATE_INTERVAL 5000 // ms (less frequent for CSV mode)
#define CALIBRATION_SAMPLES 1000   // Number of samples for initial calibration

// BLE Service and Characteristics
BLEService ppgService("12345678-1234-1234-1234-123456789abc");
BLECharacteristic ecgCharacteristic("12345678-1234-1234-1234-123456789abd", BLERead | BLENotify, 20);
BLECharacteristic ppgCharacteristic("12345678-1234-1234-1234-123456789abe", BLERead | BLENotify, 20);
BLECharacteristic heartRateCharacteristic("12345678-1234-1234-1234-123456789abf", BLERead | BLENotify, 4);

// Global objects
ArduinoTinyMLModel tinyml_model;

// Global variables for CSV playback
unsigned long last_sample_time = 0;
unsigned long last_heartbeat_time = 0;
unsigned long last_status_update = 0;
uint32_t csv_data_index = 0;           // Current position in CSV data
bool csv_loop_enabled = true;          // Loop the CSV data when finished
float calibration_buffer[CALIBRATION_SAMPLES];
bool model_initialized = false;
bool calibration_complete = false;
uint16_t calibration_index = 0;

// Performance tracking
uint32_t total_samples_processed = 0;
float average_ecg_value = 0.0;
float average_ppg_value = 0.0;
unsigned long playback_start_time = 0;

void setup() {
    Serial.begin(SERIAL_BAUD_RATE);
    
    // Wait for serial connection in debug mode
    #if ENABLE_SERIAL_DEBUG
    while (!Serial && millis() < 5000) {
        delay(10);
    }
    #endif
    
    Serial.println("Arduino Nano 33 BLE Sense - TinyML CSV ECG Playback");
    Serial.println("====================================================");
    Serial.println("Board: Arduino Nano 33 BLE Sense (ARM Cortex-M4)");
    Serial.print("CPU Frequency: ");
    Serial.print(SystemCoreClock / 1000000);
    Serial.println(" MHz");
    
    Serial.print("ECG data points loaded: ");
    Serial.println(ECG_DATA_SIZE);
    Serial.print("Playback duration: ~");
    Serial.print((float)ECG_DATA_SIZE / SAMPLING_RATE);
    Serial.println(" seconds per loop");
    
    // Initialize pins
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
    
    // Start calibration phase using CSV data
    Serial.println("Starting calibration phase using CSV ECG data...");
    Serial.print("Collecting ");
    Serial.print(CALIBRATION_SAMPLES);
    Serial.println(" samples for calibration...");
    
    digitalWrite(STATUS_LED_PIN, HIGH); // Status LED on during calibration
    
    Serial.println("Setup complete. Starting CSV ECG playback...");
    Serial.println("Format: timestamp,sample_index,ecg,ppg,heart_rate");
    
    playback_start_time = millis();
}

void loop() {
    unsigned long current_time = millis();
    
    // Check if it's time for next sample
    if (current_time - last_sample_time >= SAMPLING_INTERVAL_MS) {
        last_sample_time = current_time;
        
        // Get ECG sample from CSV data
        float ecg_sample = get_next_ecg_sample();
        
        // Handle calibration phase
        if (!calibration_complete) {
            handle_calibration(ecg_sample);
            return;
        }
        
        // Process sample through TinyML model with fallback
        float ppg_sample = tinyml_model.predict_ppg_sample(ecg_sample);
        
        // Debug: Check if model is producing valid output
        static uint32_t debug_count = 0;
        debug_count++;
        if (debug_count % 360 == 0) {  // Once per second
            Serial.print("TinyML PPG: ");
            Serial.print(ppg_sample, 4);
            Serial.print(", ECG: ");
            Serial.println(ecg_sample, 1);
        }
        
        // Fallback: If model produces unrealistic values, use simple transformation
        if (ppg_sample <= 0.05 || ppg_sample >= 2.0 || isnan(ppg_sample)) {
            static float phase = 0;
            static float prev_ecg = 0;
            static uint32_t fail_count = 0;
            
            fail_count++;
            if (fail_count % 360 == 1) {  // Print once per second
                Serial.print("WARNING: LSTM failed! PPG value: ");
                Serial.print(ppg_sample, 6);
                Serial.print(", isnan: ");
                Serial.print(isnan(ppg_sample) ? "YES" : "NO");
                Serial.print(", fail_count: ");
                Serial.println(fail_count);
            }
            
            float ecg_normalized = (ecg_sample - 350.0) / 100.0;
            float ecg_derivative = ecg_sample - prev_ecg;
            prev_ecg = ecg_sample;
            
            phase += 0.1;
            float base_ppg = 0.8 + 0.3 * sin(phase);
            float ecg_influence = 0.1 * ecg_normalized + 0.05 * ecg_derivative/10.0;
            
            ppg_sample = base_ppg + ecg_influence;
            ppg_sample = constrain(ppg_sample, 0.1, 1.5);
        } else {
            // LSTM is working! Print success message occasionally
            static uint32_t success_count = 0;
            success_count++;
            if (success_count % 360 == 1) {
                Serial.print("SUCCESS: LSTM working! PPG: ");
                Serial.println(ppg_sample, 4);
            }
        }
        
        // Output PPG signal to PWM
        tinyml_model.output_ppg_to_pwm(ppg_sample, PPG_OUTPUT_PIN);
        
        // Update running averages
        update_running_averages(ecg_sample, ppg_sample);
        
        // Handle heartbeat LED
        handle_heartbeat_led();
        
        // Send data via BLE
        send_ble_data(ecg_sample, ppg_sample);
        
        // Log data to serial with CSV index
        log_csv_data_to_serial(ecg_sample, ppg_sample);
        
        total_samples_processed++;
    }
    
    // Handle BLE connections
    handle_ble_connections();
    
    // Periodic status updates
    handle_status_updates();
    
    // Handle serial commands
    handle_serial_commands();
}

float get_next_ecg_sample() {
    // Get current ECG value from array
    float ecg_value = pgm_read_float(&ecg_data_array[csv_data_index]);
    
    // Advance to next sample
    csv_data_index++;
    
    // Check if we've reached the end of data
    if (csv_data_index >= ECG_DATA_SIZE) {
        if (csv_loop_enabled) {
            csv_data_index = 0; // Loop back to beginning
            Serial.println("CSV playback looped to beginning");
        } else {
            csv_data_index = ECG_DATA_SIZE - 1; // Stay at last sample
        }
    }
    
    return ecg_value;
}

void setup_ble_service() {
    BLE.setLocalName("TinyML-CSV");
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
    Serial.println("BLE service started. Advertising as 'TinyML-CSV'");
}

void handle_calibration(float ecg_sample) {
    if (calibration_index < CALIBRATION_SAMPLES) {
        calibration_buffer[calibration_index] = ecg_sample;
        calibration_index++;
        
        // Show calibration progress
        if (calibration_index % 100 == 0) {
            Serial.print("Calibration progress: ");
            Serial.print((calibration_index * 100) / CALIBRATION_SAMPLES);
            Serial.print("% (CSV index: ");
            Serial.print(csv_data_index);
            Serial.println(")");
        }
    } else {
        // Complete calibration
        Serial.println("Calibration complete. Processing calibration data...");
        
        tinyml_model.calibrate_input(calibration_buffer, CALIBRATION_SAMPLES);
        calibration_complete = true;
        
        digitalWrite(STATUS_LED_PIN, LOW); // Turn off status LED
        
        Serial.println("Model calibrated with CSV ECG data!");
        Serial.println("Starting real-time TinyML inference...");
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

void log_csv_data_to_serial(float ecg, float ppg) {
    #if ENABLE_SERIAL_DEBUG
    Serial.print(millis());
    Serial.print(",");
    Serial.print(csv_data_index);
    Serial.print(",");
    Serial.print(ecg, 4);
    Serial.print(",");
    Serial.print(ppg, 4);
    Serial.print(",");
    Serial.println(tinyml_model.get_heart_rate_estimate(), 1);
    #endif
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
    Serial.println("=== CSV Playback Status ===");
    Serial.print("Playback time: ");
    Serial.print((millis() - playback_start_time) / 1000);
    Serial.println(" seconds");
    
    Serial.print("CSV data position: ");
    Serial.print(csv_data_index);
    Serial.print("/");
    Serial.print(ECG_DATA_SIZE);
    Serial.print(" (");
    Serial.print((csv_data_index * 100) / ECG_DATA_SIZE);
    Serial.println("%)");
    
    Serial.print("Samples processed: ");
    Serial.println(total_samples_processed);
    
    Serial.print("Sample rate: ");
    if (total_samples_processed > 0) {
        float actual_rate = (float)total_samples_processed * 1000.0 / (millis() - playback_start_time);
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
    
    Serial.println("===========================");
}

void handle_serial_commands() {
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');
        command.trim();
        
        if (command == "status") {
            print_status_summary();
        } else if (command == "reset") {
            Serial.println("Resetting model and CSV playback...");
            tinyml_model.reset_model();
            csv_data_index = 0;
            total_samples_processed = 0;
            playback_start_time = millis();
            Serial.println("Reset complete.");
        } else if (command == "loop") {
            csv_loop_enabled = !csv_loop_enabled;
            Serial.print("CSV looping: ");
            Serial.println(csv_loop_enabled ? "ENABLED" : "DISABLED");
        } else if (command == "restart") {
            Serial.println("Restarting CSV playback from beginning...");
            csv_data_index = 0;
            playback_start_time = millis();
            total_samples_processed = 0;
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
    Serial.println("reset     - Reset model and restart CSV playback");
    Serial.println("loop      - Toggle CSV looping on/off");
    Serial.println("restart   - Restart CSV from beginning");
    Serial.println("test      - Run model self-test");
    Serial.println("help      - Show this help message");
    Serial.println("==========================");
}
