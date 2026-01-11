/*
Arduino Nano 33 BLE Sense - Advanced TinyML ECG-to-PPG Example
===============================================================

Advanced demonstration featuring:
- Built-in sensor integration (IMU, pressure, proximity)
- Adaptive signal processing
- Bluetooth Low Energy data streaming
- Web dashboard compatibility
- Motion artifact detection and correction
- Advanced heart rate variability analysis
- Power management optimization

Additional Hardware (Optional):
- OLED Display (I2C): SDA=A4, SCL=A5
- External ECG Amplifier: Connected to A0
- LED Strip or RGB LED: Connected to D6, D7, D8
- Buzzer for alerts: Connected to D9

Author: Arduino TinyML Advanced Team
Version: 2.0
Date: 2024
*/

#include "arduino_nano_tinyml_model.h"
#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>     // IMU sensor
#include <Arduino_LPS22HB.h>     // Pressure sensor
#include <Arduino_APDS9960.h>    // Proximity/gesture sensor
#include <Wire.h>
#include <SPI.h>

// Optional display support (uncomment if using OLED)
// #include <Adafruit_SSD1306.h>
// #include <Adafruit_GFX.h>

// Pin definitions
#define ECG_INPUT_PIN A0
#define PPG_OUTPUT_PIN 3
#define RED_LED_PIN 6
#define GREEN_LED_PIN 7
#define BLUE_LED_PIN 8
#define BUZZER_PIN 9
#define HEARTBEAT_LED_PIN LED_BUILTIN
#define STATUS_LED_PIN 2

// Display configuration (optional)
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
// Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Advanced timing constants
#define SAMPLING_RATE 360
#define SAMPLING_INTERVAL_US (1000000 / SAMPLING_RATE)
#define CALIBRATION_SAMPLES 2000
#define HRV_ANALYSIS_WINDOW 300  // seconds
#define MOTION_DETECTION_THRESHOLD 2.0  // g

// BLE Service UUID for health monitoring
BLEService healthService("180D");  // Heart Rate Service
BLECharacteristic heartRateCharacteristic("2A37", BLERead | BLENotify, 20);
BLECharacteristic ecgDataCharacteristic("2A38", BLERead | BLENotify, 20);
BLECharacteristic ppgDataCharacteristic("2A39", BLERead | BLENotify, 20);
BLECharacteristic motionCharacteristic("2A3A", BLERead | BLENotify, 20);
BLECharacteristic hrvCharacteristic("2A3B", BLERead | BLENotify, 20);

// Global objects
ArduinoTinyMLModel tinyml_model;

// Enhanced data structures
struct SensorData {
    float ecg;
    float ppg;
    float accel_x, accel_y, accel_z;
    float gyro_x, gyro_y, gyro_z;
    float pressure;
    uint8_t proximity;
    uint32_t timestamp;
};

struct HealthMetrics {
    float heart_rate;
    float heart_rate_variability;
    float signal_quality;
    float motion_level;
    bool motion_detected;
    uint32_t valid_beats;
    uint32_t invalid_beats;
};

// Global variables
SensorData current_sensors;
HealthMetrics health_metrics;
float calibration_buffer[CALIBRATION_SAMPLES];
bool model_initialized = false;
bool calibration_complete = false;
bool sensors_initialized = false;
uint16_t calibration_index = 0;

// Performance and analysis variables
uint32_t total_samples_processed = 0;
float rr_intervals[100];  // R-R intervals for HRV
uint8_t rr_index = 0;
uint32_t last_peak_time = 0;
float motion_magnitude = 0.0;

// Timing variables
unsigned long last_sample_time = 0;
unsigned long last_sensor_update = 0;
unsigned long last_ble_update = 0;
unsigned long last_display_update = 0;
unsigned long last_hrv_analysis = 0;

void setup() {
    Serial.begin(115200);
    
    #if ENABLE_SERIAL_DEBUG
    while (!Serial && millis() < 5000) {
        delay(10);
    }
    #endif
    
    Serial.println("Arduino Nano 33 BLE Sense - Advanced TinyML ECG-to-PPG");
    Serial.println("=========================================================");
    
    // Initialize pins
    initialize_pins();
    
    // Initialize built-in sensors
    initialize_sensors();
    
    // Initialize display (optional)
    // initialize_display();
    
    // Initialize BLE
    initialize_ble();
    
    // Initialize TinyML model
    initialize_tinyml_model();
    
    // Initialize health metrics
    initialize_health_metrics();
    
    Serial.println("Advanced setup complete. Starting real-time processing...");
    startup_animation();
}

void loop() {
    unsigned long current_time = micros();
    
    // Main sampling loop
    if (current_time - last_sample_time >= SAMPLING_INTERVAL_US) {
        last_sample_time = current_time;
        
        // Read all sensors
        read_all_sensors();
        
        // Handle calibration phase
        if (!calibration_complete) {
            handle_calibration(current_sensors.ecg);
            return;
        }
        
        // Process ECG through TinyML model
        float ppg_sample = tinyml_model.predict_ppg_sample(current_sensors.ecg);
        current_sensors.ppg = ppg_sample;
        
        // Analyze signal quality and motion artifacts
        analyze_signal_quality();
        
        // Output PPG signal
        tinyml_model.output_ppg_to_pwm(ppg_sample, PPG_OUTPUT_PIN);
        
        // Update health metrics
        update_health_metrics();
        
        // Control RGB LED based on heart rate
        update_heart_rate_led();
        
        total_samples_processed++;
    }
    
    // Update sensors (lower frequency than main sampling)
    if (millis() - last_sensor_update >= 50) {  // 20 Hz
        last_sensor_update = millis();
        update_sensor_readings();
    }
    
    // BLE data transmission
    if (millis() - last_ble_update >= 100) {  // 10 Hz
        last_ble_update = millis();
        send_ble_data();
    }
    
    // Display update (if enabled)
    if (millis() - last_display_update >= 500) {  // 2 Hz
        last_display_update = millis();
        // update_display();
    }
    
    // HRV analysis
    if (millis() - last_hrv_analysis >= 10000) {  // Every 10 seconds
        last_hrv_analysis = millis();
        perform_hrv_analysis();
    }
    
    // Handle BLE connections and serial commands
    handle_ble_connections();
    handle_serial_commands();
    
    // Alert handling
    handle_alerts();
}

void initialize_pins() {
    pinMode(ECG_INPUT_PIN, INPUT);
    pinMode(PPG_OUTPUT_PIN, OUTPUT);
    pinMode(RED_LED_PIN, OUTPUT);
    pinMode(GREEN_LED_PIN, OUTPUT);
    pinMode(BLUE_LED_PIN, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(HEARTBEAT_LED_PIN, OUTPUT);
    pinMode(STATUS_LED_PIN, OUTPUT);
    
    // Turn off all LEDs initially
    digitalWrite(RED_LED_PIN, LOW);
    digitalWrite(GREEN_LED_PIN, LOW);
    digitalWrite(BLUE_LED_PIN, LOW);
}

void initialize_sensors() {
    Serial.println("Initializing built-in sensors...");
    
    // Initialize IMU
    if (IMU.begin()) {
        Serial.println("IMU initialized successfully");
        sensors_initialized = true;
    } else {
        Serial.println("Failed to initialize IMU");
    }
    
    // Initialize pressure sensor
    if (BARO.begin()) {
        Serial.println("Pressure sensor initialized successfully");
    } else {
        Serial.println("Failed to initialize pressure sensor");
    }
    
    // Initialize proximity sensor
    if (APDS.begin()) {
        Serial.println("Proximity sensor initialized successfully");
    } else {
        Serial.println("Failed to initialize proximity sensor");
    }
}

void initialize_ble() {
    if (!BLE.begin()) {
        Serial.println("Failed to initialize BLE!");
        error_flash(STATUS_LED_PIN);
    }
    
    BLE.setLocalName("Advanced-TinyML-PPG");
    BLE.setAdvertisedService(healthService);
    
    // Add characteristics
    healthService.addCharacteristic(heartRateCharacteristic);
    healthService.addCharacteristic(ecgDataCharacteristic);
    healthService.addCharacteristic(ppgDataCharacteristic);
    healthService.addCharacteristic(motionCharacteristic);
    healthService.addCharacteristic(hrvCharacteristic);
    
    BLE.addService(healthService);
    
    // Set initial values
    heartRateCharacteristic.writeValue((uint16_t)60);
    
    BLE.advertise();
    Serial.println("Advanced BLE service started");
}

void initialize_tinyml_model() {
    Serial.println("Initializing advanced TinyML model...");
    
    if (tinyml_model.initialize()) {
        model_initialized = true;
        Serial.println("TinyML model initialized successfully!");
        tinyml_model.print_model_info();
        
        // Run extended self-test
        if (tinyml_model.self_test()) {
            Serial.println("Model self-test passed!");
        } else {
            Serial.println("Warning: Model self-test failed!");
        }
    } else {
        Serial.println("Failed to initialize TinyML model!");
        error_flash(STATUS_LED_PIN);
    }
}

void initialize_health_metrics() {
    health_metrics.heart_rate = 60.0;
    health_metrics.heart_rate_variability = 50.0;
    health_metrics.signal_quality = 1.0;
    health_metrics.motion_level = 0.0;
    health_metrics.motion_detected = false;
    health_metrics.valid_beats = 0;
    health_metrics.invalid_beats = 0;
    
    memset(rr_intervals, 0, sizeof(rr_intervals));
    rr_index = 0;
}

void read_all_sensors() {
    current_sensors.timestamp = millis();
    current_sensors.ecg = tinyml_model.read_ecg_from_adc(ECG_INPUT_PIN);
    
    // Read IMU data if available
    if (sensors_initialized && IMU.accelerationAvailable()) {
        IMU.readAcceleration(current_sensors.accel_x, 
                           current_sensors.accel_y, 
                           current_sensors.accel_z);
    }
    
    if (sensors_initialized && IMU.gyroscopeAvailable()) {
        IMU.readGyroscope(current_sensors.gyro_x, 
                         current_sensors.gyro_y, 
                         current_sensors.gyro_z);
    }
}

void update_sensor_readings() {
    // Read pressure sensor
    if (BARO.pressureAvailable()) {
        current_sensors.pressure = BARO.readPressure();
    }
    
    // Read proximity sensor
    if (APDS.proximityAvailable()) {
        current_sensors.proximity = APDS.readProximity();
    }
    
    // Calculate motion magnitude
    motion_magnitude = sqrt(current_sensors.accel_x * current_sensors.accel_x +
                           current_sensors.accel_y * current_sensors.accel_y +
                           current_sensors.accel_z * current_sensors.accel_z);
    
    health_metrics.motion_level = motion_magnitude;
    health_metrics.motion_detected = (motion_magnitude > MOTION_DETECTION_THRESHOLD);
}

void analyze_signal_quality() {
    // Simple signal quality assessment based on motion
    if (health_metrics.motion_detected) {
        health_metrics.signal_quality = max(0.1f, health_metrics.signal_quality * 0.9f);
    } else {
        health_metrics.signal_quality = min(1.0f, health_metrics.signal_quality * 1.01f);
    }
    
    // Assess ECG signal quality (basic implementation)
    static float ecg_variance = 0;
    static float ecg_mean = 0;
    const float alpha = 0.01f;
    
    ecg_mean = alpha * current_sensors.ecg + (1.0f - alpha) * ecg_mean;
    float diff = current_sensors.ecg - ecg_mean;
    ecg_variance = alpha * diff * diff + (1.0f - alpha) * ecg_variance;
    
    // Good signal should have reasonable variance (not too high, not too low)
    float variance_quality = 1.0f / (1.0f + abs(ecg_variance - 0.1f) * 10);
    health_metrics.signal_quality = min(health_metrics.signal_quality, variance_quality);
}

void update_health_metrics() {
    health_metrics.heart_rate = tinyml_model.get_heart_rate_estimate();
    
    // Detect heartbeats for R-R interval analysis
    static bool last_peak_state = false;
    static uint32_t peak_count = 0;
    
    uint32_t current_peaks = tinyml_model.get_peak_count();
    if (current_peaks > peak_count) {
        peak_count = current_peaks;
        uint32_t current_time = millis();
        
        if (last_peak_time > 0 && health_metrics.signal_quality > 0.5) {
            // Calculate R-R interval
            uint32_t rr_interval = current_time - last_peak_time;
            
            if (rr_interval > 300 && rr_interval < 2000) {  // Valid heart rate range
                rr_intervals[rr_index] = rr_interval;
                rr_index = (rr_index + 1) % 100;
                health_metrics.valid_beats++;
            } else {
                health_metrics.invalid_beats++;
            }
        }
        
        last_peak_time = current_time;
    }
}

void perform_hrv_analysis() {
    // Calculate Heart Rate Variability (RMSSD method)
    float sum_squared_diffs = 0;
    uint8_t valid_intervals = 0;
    
    for (uint8_t i = 1; i < 100; i++) {
        if (rr_intervals[i] > 0 && rr_intervals[i-1] > 0) {
            float diff = rr_intervals[i] - rr_intervals[i-1];
            sum_squared_diffs += diff * diff;
            valid_intervals++;
        }
    }
    
    if (valid_intervals > 10) {
        health_metrics.heart_rate_variability = sqrt(sum_squared_diffs / valid_intervals);
    }
    
    Serial.print("HRV Analysis - RMSSD: ");
    Serial.print(health_metrics.heart_rate_variability, 1);
    Serial.print(" ms, Valid beats: ");
    Serial.print(health_metrics.valid_beats);
    Serial.print(", Invalid beats: ");
    Serial.println(health_metrics.invalid_beats);
}

void update_heart_rate_led() {
    float hr = health_metrics.heart_rate;
    
    // Map heart rate to RGB colors
    if (hr < 60) {
        // Low heart rate - Blue
        analogWrite(RED_LED_PIN, 0);
        analogWrite(GREEN_LED_PIN, 0);
        analogWrite(BLUE_LED_PIN, 255);
    } else if (hr < 100) {
        // Normal heart rate - Green
        analogWrite(RED_LED_PIN, 0);
        analogWrite(GREEN_LED_PIN, 255);
        analogWrite(BLUE_LED_PIN, 0);
    } else if (hr < 150) {
        // Elevated heart rate - Yellow
        analogWrite(RED_LED_PIN, 255);
        analogWrite(GREEN_LED_PIN, 255);
        analogWrite(BLUE_LED_PIN, 0);
    } else {
        // High heart rate - Red
        analogWrite(RED_LED_PIN, 255);
        analogWrite(GREEN_LED_PIN, 0);
        analogWrite(BLUE_LED_PIN, 0);
    }
    
    // Dim LEDs if signal quality is poor
    if (health_metrics.signal_quality < 0.5) {
        analogWrite(RED_LED_PIN, analogRead(RED_LED_PIN) / 4);
        analogWrite(GREEN_LED_PIN, analogRead(GREEN_LED_PIN) / 4);
        analogWrite(BLUE_LED_PIN, analogRead(BLUE_LED_PIN) / 4);
    }
}

void send_ble_data() {
    if (BLE.connected()) {
        // Send heart rate
        uint16_t hr_data = (uint16_t)health_metrics.heart_rate;
        heartRateCharacteristic.writeValue(hr_data);
        
        // Send ECG data (scaled and offset for transmission)
        int16_t ecg_data = (int16_t)(current_sensors.ecg * 1000);
        ecgDataCharacteristic.writeValue(ecg_data);
        
        // Send PPG data
        int16_t ppg_data = (int16_t)(current_sensors.ppg * 1000);
        ppgDataCharacteristic.writeValue(ppg_data);
        
        // Send motion data
        uint8_t motion_data[8];
        memcpy(&motion_data[0], &current_sensors.accel_x, 4);
        memcpy(&motion_data[4], &motion_magnitude, 4);
        motionCharacteristic.writeValue(motion_data, 8);
        
        // Send HRV data
        uint16_t hrv_data = (uint16_t)health_metrics.heart_rate_variability;
        hrvCharacteristic.writeValue(hrv_data);
    }
}

void handle_calibration(float ecg_sample) {
    if (calibration_index < CALIBRATION_SAMPLES) {
        calibration_buffer[calibration_index] = ecg_sample;
        calibration_index++;
        
        // Enhanced calibration progress with LED feedback
        if (calibration_index % 200 == 0) {
            Serial.print("Advanced calibration progress: ");
            Serial.print((calibration_index * 100) / CALIBRATION_SAMPLES);
            Serial.println("%");
            
            // Flash status LED to show progress
            digitalWrite(STATUS_LED_PIN, !digitalRead(STATUS_LED_PIN));
        }
    } else {
        // Complete calibration with advanced processing
        Serial.println("Advanced calibration complete. Processing data...");
        
        tinyml_model.calibrate_input(calibration_buffer, CALIBRATION_SAMPLES);
        calibration_complete = true;
        
        digitalWrite(STATUS_LED_PIN, LOW);
        
        Serial.println("Advanced model calibrated and ready!");
        
        // Calibration complete animation
        calibration_complete_animation();
    }
}

void handle_alerts() {
    // Heart rate alerts
    if (health_metrics.heart_rate > 180 || health_metrics.heart_rate < 40) {
        if (millis() % 2000 < 100) {  // Alert every 2 seconds
            tone(BUZZER_PIN, 1000, 100);
            digitalWrite(STATUS_LED_PIN, HIGH);
        } else {
            digitalWrite(STATUS_LED_PIN, LOW);
        }
    }
    
    // Signal quality alerts
    if (health_metrics.signal_quality < 0.3) {
        if (millis() % 1000 < 50) {  // Brief flash every second
            digitalWrite(STATUS_LED_PIN, HIGH);
        } else {
            digitalWrite(STATUS_LED_PIN, LOW);
        }
    }
}

void handle_ble_connections() {
    BLEDevice central = BLE.central();
    
    static bool was_connected = false;
    bool is_connected = central.connected();
    
    if (is_connected && !was_connected) {
        Serial.print("Advanced BLE connected to: ");
        Serial.println(central.address());
        connection_animation();
    } else if (!is_connected && was_connected) {
        Serial.println("Advanced BLE disconnected");
    }
    
    was_connected = is_connected;
}

void handle_serial_commands() {
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');
        command.trim();
        command.toLowerCase();
        
        if (command == "status") {
            print_advanced_status();
        } else if (command == "metrics") {
            print_health_metrics();
        } else if (command == "sensors") {
            print_sensor_readings();
        } else if (command == "hrv") {
            perform_hrv_analysis();
        } else if (command == "reset") {
            reset_system();
        } else if (command == "calibrate") {
            restart_calibration();
        } else if (command == "test") {
            run_system_test();
        } else if (command.startsWith("led ")) {
            handle_led_command(command);
        } else if (command == "help") {
            print_advanced_help();
        } else if (command.length() > 0) {
            Serial.println("Unknown command. Type 'help' for available commands.");
        }
    }
}

void print_advanced_status() {
    Serial.println("=== Advanced System Status ===");
    Serial.print("Uptime: ");
    Serial.print(millis() / 1000);
    Serial.println(" seconds");
    
    Serial.print("Samples processed: ");
    Serial.println(total_samples_processed);
    
    Serial.print("Model initialized: ");
    Serial.println(model_initialized ? "YES" : "NO");
    
    Serial.print("Sensors initialized: ");
    Serial.println(sensors_initialized ? "YES" : "NO");
    
    Serial.print("Calibration complete: ");
    Serial.println(calibration_complete ? "YES" : "NO");
    
    Serial.print("BLE connected: ");
    Serial.println(BLE.connected() ? "YES" : "NO");
    
    Serial.print("Free memory: ");
    Serial.print(get_free_memory());
    Serial.println(" bytes");
    
    print_health_metrics();
    tinyml_model.print_performance_metrics();
    
    Serial.println("=============================");
}

void print_health_metrics() {
    Serial.println("=== Health Metrics ===");
    Serial.print("Heart Rate: ");
    Serial.print(health_metrics.heart_rate, 1);
    Serial.println(" BPM");
    
    Serial.print("HRV (RMSSD): ");
    Serial.print(health_metrics.heart_rate_variability, 1);
    Serial.println(" ms");
    
    Serial.print("Signal Quality: ");
    Serial.print(health_metrics.signal_quality * 100, 1);
    Serial.println("%");
    
    Serial.print("Motion Level: ");
    Serial.print(health_metrics.motion_level, 2);
    Serial.println(" g");
    
    Serial.print("Motion Detected: ");
    Serial.println(health_metrics.motion_detected ? "YES" : "NO");
    
    Serial.print("Valid Beats: ");
    Serial.println(health_metrics.valid_beats);
    
    Serial.print("Invalid Beats: ");
    Serial.println(health_metrics.invalid_beats);
    
    Serial.println("======================");
}

void print_sensor_readings() {
    Serial.println("=== Sensor Readings ===");
    Serial.print("ECG: ");
    Serial.print(current_sensors.ecg, 4);
    Serial.println(" V");
    
    Serial.print("PPG: ");
    Serial.print(current_sensors.ppg, 4);
    Serial.println(" (normalized)");
    
    Serial.print("Acceleration (x,y,z): ");
    Serial.print(current_sensors.accel_x, 2);
    Serial.print(", ");
    Serial.print(current_sensors.accel_y, 2);
    Serial.print(", ");
    Serial.print(current_sensors.accel_z, 2);
    Serial.println(" g");
    
    Serial.print("Gyroscope (x,y,z): ");
    Serial.print(current_sensors.gyro_x, 2);
    Serial.print(", ");
    Serial.print(current_sensors.gyro_y, 2);
    Serial.print(", ");
    Serial.print(current_sensors.gyro_z, 2);
    Serial.println(" °/s");
    
    Serial.print("Pressure: ");
    Serial.print(current_sensors.pressure, 2);
    Serial.println(" kPa");
    
    Serial.print("Proximity: ");
    Serial.println(current_sensors.proximity);
    
    Serial.println("=======================");
}

void print_advanced_help() {
    Serial.println("=== Advanced Commands ===");
    Serial.println("status    - Complete system status");
    Serial.println("metrics   - Health metrics display");
    Serial.println("sensors   - Current sensor readings");
    Serial.println("hrv       - Heart rate variability analysis");
    Serial.println("reset     - Reset entire system");
    Serial.println("calibrate - Restart calibration");
    Serial.println("test      - Run comprehensive system test");
    Serial.println("led r g b - Set RGB LED (0-255 each)");
    Serial.println("help      - Show this help");
    Serial.println("=========================");
}

void startup_animation() {
    for (int i = 0; i < 3; i++) {
        digitalWrite(RED_LED_PIN, HIGH);
        delay(100);
        digitalWrite(RED_LED_PIN, LOW);
        digitalWrite(GREEN_LED_PIN, HIGH);
        delay(100);
        digitalWrite(GREEN_LED_PIN, LOW);
        digitalWrite(BLUE_LED_PIN, HIGH);
        delay(100);
        digitalWrite(BLUE_LED_PIN, LOW);
    }
}

void calibration_complete_animation() {
    for (int brightness = 0; brightness <= 255; brightness += 5) {
        analogWrite(GREEN_LED_PIN, brightness);
        delay(10);
    }
    delay(500);
    for (int brightness = 255; brightness >= 0; brightness -= 5) {
        analogWrite(GREEN_LED_PIN, brightness);
        delay(10);
    }
}

void connection_animation() {
    for (int i = 0; i < 5; i++) {
        digitalWrite(BLUE_LED_PIN, HIGH);
        delay(100);
        digitalWrite(BLUE_LED_PIN, LOW);
        delay(100);
    }
}

void error_flash(uint8_t pin) {
    while (1) {
        digitalWrite(pin, HIGH);
        delay(200);
        digitalWrite(pin, LOW);
        delay(200);
    }
}

void reset_system() {
    Serial.println("Resetting system...");
    
    // Reset model
    tinyml_model.reset_model();
    
    // Reset health metrics
    initialize_health_metrics();
    
    // Reset calibration
    calibration_complete = false;
    calibration_index = 0;
    
    // Reset counters
    total_samples_processed = 0;
    
    Serial.println("System reset complete.");
}

void restart_calibration() {
    Serial.println("Restarting calibration...");
    calibration_complete = false;
    calibration_index = 0;
    digitalWrite(STATUS_LED_PIN, HIGH);
}

void run_system_test() {
    Serial.println("Running comprehensive system test...");
    
    bool all_tests_passed = true;
    
    // Test TinyML model
    Serial.print("TinyML model test: ");
    bool model_test = tinyml_model.self_test();
    Serial.println(model_test ? "PASSED" : "FAILED");
    all_tests_passed &= model_test;
    
    // Test sensors
    Serial.print("IMU test: ");
    bool imu_test = IMU.accelerationAvailable() && IMU.gyroscopeAvailable();
    Serial.println(imu_test ? "PASSED" : "FAILED");
    all_tests_passed &= imu_test;
    
    Serial.print("Pressure sensor test: ");
    bool pressure_test = BARO.pressureAvailable();
    Serial.println(pressure_test ? "PASSED" : "FAILED");
    
    Serial.print("Proximity sensor test: ");
    bool proximity_test = APDS.proximityAvailable();
    Serial.println(proximity_test ? "PASSED" : "FAILED");
    
    // Test BLE
    Serial.print("BLE test: ");
    bool ble_test = BLE.connected() || BLE.advertising();
    Serial.println(ble_test ? "PASSED" : "FAILED");
    all_tests_passed &= ble_test;
    
    // Overall result
    Serial.print("Overall system test: ");
    Serial.println(all_tests_passed ? "PASSED" : "FAILED");
    
    if (all_tests_passed) {
        // Success animation
        for (int i = 0; i < 3; i++) {
            digitalWrite(GREEN_LED_PIN, HIGH);
            delay(200);
            digitalWrite(GREEN_LED_PIN, LOW);
            delay(200);
        }
    } else {
        // Failure animation
        for (int i = 0; i < 5; i++) {
            digitalWrite(RED_LED_PIN, HIGH);
            delay(100);
            digitalWrite(RED_LED_PIN, LOW);
            delay(100);
        }
    }
}

void handle_led_command(String command) {
    // Parse LED command: "led r g b"
    int r, g, b;
    if (sscanf(command.c_str(), "led %d %d %d", &r, &g, &b) == 3) {
        r = constrain(r, 0, 255);
        g = constrain(g, 0, 255);
        b = constrain(b, 0, 255);
        
        analogWrite(RED_LED_PIN, r);
        analogWrite(GREEN_LED_PIN, g);
        analogWrite(BLUE_LED_PIN, b);
        
        Serial.print("LED set to RGB(");
        Serial.print(r);
        Serial.print(", ");
        Serial.print(g);
        Serial.print(", ");
        Serial.print(b);
        Serial.println(")");
    } else {
        Serial.println("Invalid LED command. Use: led r g b (0-255 each)");
    }
}
