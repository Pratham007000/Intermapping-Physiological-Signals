/*
Arduino Nano 33 BLE Sense - Simple TinyML Demo
===============================================

This is a simplified version of the TinyML ECG-to-PPG converter
for testing and demonstration purposes. Use this if you want to
verify the basic functionality before using the full version.

Hardware Requirements:
- Arduino Nano 33 BLE Sense
- ECG sensor connected to A0 (or use potentiometer for testing)
- LED on pin 2 for status indication

Library Dependencies:
- ArduinoBLE (install via Library Manager)

Board Selection:
- Tools > Board > Arduino Mbed OS Nano Boards > Arduino Nano 33 BLE
*/

// Board compatibility check
#if !defined(ARDUINO_ARCH_MBED_NANO) && !defined(ARDUINO_ARCH_MBED)
  #error "This code is designed specifically for Arduino Nano 33 BLE Sense. Please select the correct board in Tools > Board."
#endif

#include <ArduinoBLE.h>

// Pin definitions
#define ECG_INPUT_PIN A0
#define STATUS_LED_PIN 2
#define HEARTBEAT_LED_PIN LED_BUILTIN

// Timing constants
#define SAMPLE_RATE_HZ 100    // Reduced for demo
#define SAMPLE_INTERVAL_MS (1000 / SAMPLE_RATE_HZ)

// Simple moving average filter
#define FILTER_SIZE 5
float ecg_buffer[FILTER_SIZE] = {0};
float ppg_buffer[FILTER_SIZE] = {0};
int buffer_index = 0;

// Heart rate estimation
float heart_rate_estimate = 60.0;
unsigned long last_peak_time = 0;
float peak_threshold = 0.5;

// Performance tracking
unsigned long last_sample_time = 0;
unsigned long samples_processed = 0;

// BLE setup
BLEService demoService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLEFloatCharacteristic ecgCharacteristic("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify);
BLEFloatCharacteristic ppgCharacteristic("19B10002-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify);
BLEFloatCharacteristic heartRateCharacteristic("19B10003-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify);

void setup() {
  Serial.begin(115200);
  
  // Wait for serial connection (max 5 seconds)
  unsigned long start_time = millis();
  while (!Serial && (millis() - start_time < 5000)) {
    delay(10);
  }
  
  Serial.println("Arduino Nano 33 BLE Sense - Simple TinyML Demo");
  Serial.println("===============================================");
  Serial.print("Board: Arduino Nano 33 BLE Sense (ARM Cortex-M4 @ ");
  Serial.print(SystemCoreClock / 1000000);
  Serial.println(" MHz)");
  
  // Initialize pins
  pinMode(ECG_INPUT_PIN, INPUT);
  pinMode(STATUS_LED_PIN, OUTPUT);
  pinMode(HEARTBEAT_LED_PIN, OUTPUT);
  
  // Flash status LED to indicate startup
  for (int i = 0; i < 3; i++) {
    digitalWrite(STATUS_LED_PIN, HIGH);
    delay(200);
    digitalWrite(STATUS_LED_PIN, LOW);
    delay(200);
  }
  
  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1) {
      digitalWrite(STATUS_LED_PIN, HIGH);
      delay(100);
      digitalWrite(STATUS_LED_PIN, LOW);
      delay(100);
    }
  }
  
  // Set up BLE service
  BLE.setLocalName("TinyML-Demo");
  BLE.setAdvertisedService(demoService);
  
  demoService.addCharacteristic(ecgCharacteristic);
  demoService.addCharacteristic(ppgCharacteristic);
  demoService.addCharacteristic(heartRateCharacteristic);
  
  BLE.addService(demoService);
  
  // Set initial values
  ecgCharacteristic.writeValue(0.0);
  ppgCharacteristic.writeValue(0.0);
  heartRateCharacteristic.writeValue(heart_rate_estimate);
  
  BLE.advertise();
  
  Serial.println("BLE initialized. Device name: TinyML-Demo");
  Serial.println("Connect with a BLE app to monitor data.");
  Serial.println();
  Serial.println("Demo started! Data format: timestamp,ecg,ppg,heart_rate");
  Serial.println("Connect A0 to ECG sensor or potentiometer for testing.");
  Serial.println();
  
  digitalWrite(STATUS_LED_PIN, HIGH); // Status LED on when ready
}

void loop() {
  unsigned long current_time = millis();
  
  // Sample at fixed rate
  if (current_time - last_sample_time >= SAMPLE_INTERVAL_MS) {
    last_sample_time = current_time;
    
    // Read ECG sample from A0
    float ecg_raw = readECGSample();
    
    // Apply simple filtering
    float ecg_filtered = applyFilter(ecg_raw, ecg_buffer);
    
    // Simple ECG-to-PPG conversion (demo algorithm)
    float ppg_estimate = convertECGtoPPG(ecg_filtered);
    
    // Apply filtering to PPG
    float ppg_filtered = applyFilter(ppg_estimate, ppg_buffer);
    
    // Detect peaks and estimate heart rate
    detectPeakAndUpdateHeartRate(ppg_filtered);
    
    // Output to serial
    Serial.print(current_time);
    Serial.print(",");
    Serial.print(ecg_filtered, 4);
    Serial.print(",");
    Serial.print(ppg_filtered, 4);
    Serial.print(",");
    Serial.println(heart_rate_estimate, 1);
    
    // Send via BLE if connected
    if (BLE.connected()) {
      ecgCharacteristic.writeValue(ecg_filtered);
      ppgCharacteristic.writeValue(ppg_filtered);
      
      // Send heart rate less frequently
      if (samples_processed % 10 == 0) {
        heartRateCharacteristic.writeValue(heart_rate_estimate);
      }
    }
    
    // Update heartbeat LED
    updateHeartbeatLED();
    
    samples_processed++;
    
    // Print status every 10 seconds
    if (samples_processed % (SAMPLE_RATE_HZ * 10) == 0) {
      printStatus();
    }
  }
  
  // Handle BLE events
  BLE.poll();
  
  // Handle serial commands
  handleSerialCommands();
}

float readECGSample() {
  // Read analog value (0-1023)
  int adc_value = analogRead(ECG_INPUT_PIN);
  
  // Convert to voltage (0-3.3V)
  float voltage = (float)adc_value * 3.3 / 1023.0;
  
  // Center around 1.65V and scale to ±1.65V range
  float ecg_signal = voltage - 1.65;
  
  return ecg_signal;
}

float applyFilter(float new_sample, float* buffer) {
  // Add new sample to circular buffer
  buffer[buffer_index] = new_sample;
  buffer_index = (buffer_index + 1) % FILTER_SIZE;
  
  // Calculate moving average
  float sum = 0;
  for (int i = 0; i < FILTER_SIZE; i++) {
    sum += buffer[i];
  }
  
  return sum / FILTER_SIZE;
}

float convertECGtoPPG(float ecg_sample) {
  // Simple demo conversion: delayed and inverted ECG with some scaling
  // In reality, this would be a complex LSTM model
  
  static float delayed_ecg[20] = {0}; // Simple delay line
  static int delay_index = 0;
  
  // Store current sample
  delayed_ecg[delay_index] = ecg_sample;
  delay_index = (delay_index + 1) % 20;
  
  // Get delayed sample (simulates ECG-PPG delay)
  int delayed_idx = (delay_index + 10) % 20; // 10-sample delay
  float delayed_sample = delayed_ecg[delayed_idx];
  
  // Simple transformation: scale and add some non-linearity
  float ppg_estimate = -0.8 * delayed_sample + 0.2 * delayed_sample * delayed_sample;
  
  // Add some baseline variation
  ppg_estimate += 0.1 * sin(2 * PI * millis() / 10000.0);
  
  return ppg_estimate;
}

void detectPeakAndUpdateHeartRate(float ppg_sample) {
  static float prev_sample = 0;
  static float prev_prev_sample = 0;
  static unsigned long peak_times[5] = {0}; // Store last 5 peak times
  static int peak_count = 0;
  
  // Simple peak detection: current > previous and previous > previous-previous
  bool is_peak = (ppg_sample > prev_sample) && 
                 (prev_sample > prev_prev_sample) && 
                 (ppg_sample > peak_threshold);
  
  if (is_peak) {
    unsigned long current_time = millis();
    
    // Avoid duplicate peaks (minimum 300ms between peaks = 200 BPM max)
    if (current_time - last_peak_time > 300) {
      last_peak_time = current_time;
      
      // Store peak time in circular buffer
      peak_times[peak_count % 5] = current_time;
      peak_count++;
      
      // Calculate heart rate from last few peaks
      if (peak_count >= 3) {
        int valid_peaks = min(5, peak_count);
        unsigned long time_span = current_time - peak_times[(peak_count - valid_peaks) % 5];
        
        if (time_span > 0) {
          float calculated_hr = (60000.0 * (valid_peaks - 1)) / time_span;
          
          // Smooth the heart rate estimate
          heart_rate_estimate = 0.7 * heart_rate_estimate + 0.3 * calculated_hr;
          
          // Constrain to reasonable range
          heart_rate_estimate = constrain(heart_rate_estimate, 40, 200);
        }
      }
    }
  }
  
  // Update previous samples
  prev_prev_sample = prev_sample;
  prev_sample = ppg_sample;
}

void updateHeartbeatLED() {
  // Flash heartbeat LED based on estimated heart rate
  static unsigned long last_heartbeat_flash = 0;
  
  unsigned long beat_interval = (unsigned long)(60000.0 / heart_rate_estimate);
  unsigned long current_time = millis();
  
  if (current_time - last_heartbeat_flash >= beat_interval) {
    digitalWrite(HEARTBEAT_LED_PIN, HIGH);
    last_heartbeat_flash = current_time;
  }
  
  // Turn off LED after 100ms
  static unsigned long led_on_time = 0;
  if (digitalRead(HEARTBEAT_LED_PIN) == HIGH) {
    if (led_on_time == 0) {
      led_on_time = current_time;
    } else if (current_time - led_on_time >= 100) {
      digitalWrite(HEARTBEAT_LED_PIN, LOW);
      led_on_time = 0;
    }
  }
}

void printStatus() {
  Serial.println();
  Serial.println("=== Status ===");
  Serial.print("Uptime: ");
  Serial.print(millis() / 1000);
  Serial.println(" seconds");
  Serial.print("Samples processed: ");
  Serial.println(samples_processed);
  Serial.print("Sample rate: ");
  Serial.print(SAMPLE_RATE_HZ);
  Serial.println(" Hz");
  Serial.print("Heart rate estimate: ");
  Serial.print(heart_rate_estimate, 1);
  Serial.println(" BPM");
  Serial.print("BLE status: ");
  Serial.println(BLE.connected() ? "Connected" : "Advertising");
  Serial.print("Free memory estimate: ~");
  Serial.print(200000); // Rough estimate for Nano 33 BLE
  Serial.println(" bytes");
  Serial.println("==============");
  Serial.println();
}

void handleSerialCommands() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "status") {
      printStatus();
    } else if (command == "reset") {
      Serial.println("Resetting counters...");
      samples_processed = 0;
      heart_rate_estimate = 60.0;
      buffer_index = 0;
      // Clear buffers
      for (int i = 0; i < FILTER_SIZE; i++) {
        ecg_buffer[i] = 0;
        ppg_buffer[i] = 0;
      }
      Serial.println("Reset complete.");
    } else if (command == "help") {
      Serial.println("Available commands:");
      Serial.println("  status - Show current status");
      Serial.println("  reset  - Reset all counters and buffers");
      Serial.println("  help   - Show this help message");
    } else if (command.length() > 0) {
      Serial.print("Unknown command: ");
      Serial.println(command);
      Serial.println("Type 'help' for available commands.");
    }
  }
}
