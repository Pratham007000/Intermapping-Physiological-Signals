/*
Arduino Nano 33 BLE Sense TinyML ECG-to-PPG Model Implementation
================================================================

Complete implementation of LSTM-based ECG-to-PPG conversion model
optimized for Arduino Nano 33 BLE Sense.
*/

#include "arduino_nano_tinyml_model.h"

// Constructor
ArduinoTinyMLModel::ArduinoTinyMLModel() {
    // Initialize model parameters
    memset(&model, 0, sizeof(TinyMLModel));
    memset(&signal_processor, 0, sizeof(SignalProcessor));
    
    // Set default values
    model.input_mean = 0;
    model.input_std = FIXED_POINT_SCALE;
    model.output_bias = 0;
    model.buffer_index = 0;
    model.min_inference_time_us = UINT32_MAX;
    
    signal_processor.sampling_rate = DEFAULT_SAMPLING_RATE;
    signal_processor.heart_rate_estimate = 60.0; // Default 60 BPM
}

// Destructor
ArduinoTinyMLModel::~ArduinoTinyMLModel() {
    // Nothing to clean up for now
}

// Initialize the model
bool ArduinoTinyMLModel::initialize() {
    #if ENABLE_SERIAL_DEBUG
    Serial.println("Initializing Arduino TinyML Model...");
    #endif
    
    // Initialize matrices
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        matrix_init(&model.lstm_layers[layer].Wf, HIDDEN_SIZE, INPUT_SIZE + HIDDEN_SIZE);
        matrix_init(&model.lstm_layers[layer].Wi, HIDDEN_SIZE, INPUT_SIZE + HIDDEN_SIZE);
        matrix_init(&model.lstm_layers[layer].Wo, HIDDEN_SIZE, INPUT_SIZE + HIDDEN_SIZE);
        matrix_init(&model.lstm_layers[layer].Wg, HIDDEN_SIZE, INPUT_SIZE + HIDDEN_SIZE);
        
        model.lstm_layers[layer].input_size = INPUT_SIZE;
        model.lstm_layers[layer].hidden_size = HIDDEN_SIZE;
        
        reset_lstm_state(&model.lstm_layers[layer]);
    }
    
    // Initialize output layer weights
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        model.output_weights[i] = FIXED_POINT_SCALE / HIDDEN_SIZE;
    }
    
    // Load model weights
    if (!load_model_weights()) {
        #if ENABLE_SERIAL_DEBUG
        Serial.println("Failed to load model weights!");
        #endif
        return false;
    }
    
    #if ENABLE_SERIAL_DEBUG
    Serial.print("Model initialized. Memory usage: ");
    Serial.print(get_memory_usage());
    Serial.println(" bytes");
    #endif
    
    return true;
}

// Load pre-trained model weights
bool ArduinoTinyMLModel::load_model_weights() {
    // Copy weights from PROGMEM to SRAM for faster access
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE + HIDDEN_SIZE; j++) {
            int idx = i * (INPUT_SIZE + HIDDEN_SIZE) + j;
            model.lstm_layers[0].Wf.data[idx] = model_weights_wf[i][j];
            model.lstm_layers[0].Wi.data[idx] = model_weights_wi[i][j];
            model.lstm_layers[0].Wo.data[idx] = model_weights_wo[i][j];
            model.lstm_layers[0].Wg.data[idx] = model_weights_wg[i][j];
        }
        
        model.lstm_layers[0].bf[i] = model_bias_f[i];
        model.lstm_layers[0].bi[i] = model_bias_i[i];
        model.lstm_layers[0].bo[i] = model_bias_o[i];
        model.lstm_layers[0].bg[i] = model_bias_g[i];
        
        model.output_weights[i] = output_layer_weights[i];
    }
    
    model.output_bias = output_layer_bias;
    
    return true;
}

// Main prediction function
float ArduinoTinyMLModel::predict_ppg_sample(float ecg_sample) {
    uint32_t start_time = micros();
    
    // Preprocess input
    int32_t processed_input = preprocess_ecg_sample(ecg_sample);
    
    // Debug preprocessed input
    static uint32_t debug_count = 0;
    debug_count++;
    if (debug_count % 720 == 0) {  // Every 2 seconds
        #if ENABLE_SERIAL_DEBUG
        Serial.print("LSTM Debug - ECG: ");
        Serial.print(ecg_sample, 1);
        Serial.print(", Processed: ");
        Serial.println(processed_input);
        #endif
    }
    
    // Add to sequence buffer
    model.sequence_buffer[model.buffer_index] = processed_input;
    model.buffer_index = (model.buffer_index + 1) % SEQUENCE_LENGTH;
    
    // Forward pass through LSTM layers
    int32_t layer_input[INPUT_SIZE] = {processed_input};
    int32_t layer_output[HIDDEN_SIZE];
    
    // Initialize layer output to prevent garbage values
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        layer_output[i] = 0;
    }
    
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        lstm_forward(&model.lstm_layers[layer], layer_input, layer_output);
        
        // Copy output to input for next layer (only copy what we need)
        if (layer < NUM_LAYERS - 1) {
            for (int i = 0; i < HIDDEN_SIZE && i < INPUT_SIZE; i++) {
                layer_input[i] = layer_output[i];
            }
        }
    }
    
    // Debug LSTM output
    if (debug_count % 720 == 1) {
        #if ENABLE_SERIAL_DEBUG
        Serial.print("LSTM Output[0-3]: ");
        for (int i = 0; i < 4 && i < HIDDEN_SIZE; i++) {
            Serial.print(layer_output[i]);
            Serial.print(" ");
        }
        Serial.println();
        #endif
    }
    
    // Output layer (linear transformation)
    int64_t output_sum = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        output_sum += ((int64_t)layer_output[i] * model.output_weights[i]) / FIXED_POINT_SCALE;
    }
    output_sum += model.output_bias;
    
    // Debug final sum
    if (debug_count % 720 == 2) {
        #if ENABLE_SERIAL_DEBUG
        Serial.print("Output sum: ");
        Serial.print((long)output_sum);
        Serial.print(", Bias: ");
        Serial.println(model.output_bias);
        #endif
    }
    
    // Convert to int32_t with saturation
    int32_t raw_ppg = constrain_int32((int32_t)output_sum, MIN_INT32, MAX_INT32);
    
    // Postprocess output
    float ppg_output = postprocess_ppg_sample(raw_ppg);
    
    // Apply signal processing
    ppg_output = apply_moving_average_filter(ppg_output);
    
    // Detect peaks for heart rate estimation
    if (detect_peak(ppg_output)) {
        signal_processor.heart_rate_estimate = estimate_heart_rate();
    }
    
    // Update performance metrics
    uint32_t inference_time = micros() - start_time;
    update_performance_metrics(inference_time);
    
    return ppg_output;
}

// LSTM forward pass
void ArduinoTinyMLModel::lstm_forward(LSTMCell* cell, const int32_t* input, int32_t* output) {
    // Create combined input [input, hidden_state]
    int32_t combined_input[INPUT_SIZE + HIDDEN_SIZE];
    
    for (int i = 0; i < cell->input_size; i++) {
        combined_input[i] = input[i];
    }
    for (int i = 0; i < cell->hidden_size; i++) {
        combined_input[cell->input_size + i] = cell->state.h[i];
    }
    
    // Compute gates
    int32_t f_gate[HIDDEN_SIZE], i_gate[HIDDEN_SIZE], o_gate[HIDDEN_SIZE], g_gate[HIDDEN_SIZE];
    
    compute_gate(combined_input, &cell->Wf, cell->bf, f_gate, sigmoid_fixed);
    compute_gate(combined_input, &cell->Wi, cell->bi, i_gate, sigmoid_fixed);
    compute_gate(combined_input, &cell->Wo, cell->bo, o_gate, sigmoid_fixed);
    compute_gate(combined_input, &cell->Wg, cell->bg, g_gate, tanh_fixed);
    
    // Update cell state: c_t = f_t * c_{t-1} + i_t * g_t
    for (int i = 0; i < cell->hidden_size; i++) {
        int64_t forget_term = ((int64_t)f_gate[i] * cell->state.c[i]) / FIXED_POINT_SCALE;
        int64_t input_term = ((int64_t)i_gate[i] * g_gate[i]) / FIXED_POINT_SCALE;
        cell->state.c[i] = constrain_int32((int32_t)(forget_term + input_term), MIN_INT32, MAX_INT32);
    }
    
    // Update hidden state: h_t = o_t * tanh(c_t)
    for (int i = 0; i < cell->hidden_size; i++) {
        int32_t cell_tanh = tanh_fixed(cell->state.c[i]);
        int64_t hidden_output = ((int64_t)o_gate[i] * cell_tanh) / FIXED_POINT_SCALE;
        cell->state.h[i] = constrain_int32((int32_t)hidden_output, MIN_INT32, MAX_INT32);
        output[i] = cell->state.h[i];
    }
}

// Compute gate values
void ArduinoTinyMLModel::compute_gate(const int32_t* input, const Matrix* weights, 
                                     const int32_t* bias, int32_t* output, 
                                     int32_t (*activation)(int32_t)) {
    for (int i = 0; i < weights->rows; i++) {
        int64_t sum = 0;
        for (int j = 0; j < weights->cols; j++) {
            int idx = i * weights->cols + j;
            sum += ((int64_t)weights->data[idx] * input[j]);
        }
        sum = sum / FIXED_POINT_SCALE + bias[i];
        output[i] = activation(constrain_int32((int32_t)sum, MIN_INT32, MAX_INT32));
    }
}

// Activation functions (static for function pointers)
int32_t ArduinoTinyMLModel::tanh_fixed(int32_t x) {
    // Fast tanh approximation: tanh(x) ≈ x / (1 + |x|) for |x| < 2
    if (x > 2 * FIXED_POINT_SCALE) {
        return FIXED_POINT_SCALE;
    } else if (x < -2 * FIXED_POINT_SCALE) {
        return -FIXED_POINT_SCALE;
    } else {
        int32_t abs_x = abs(x);
        return (x * FIXED_POINT_SCALE) / (FIXED_POINT_SCALE + abs_x);
    }
}

int32_t ArduinoTinyMLModel::sigmoid_fixed(int32_t x) {
    // Fast sigmoid: sigmoid(x) ≈ 0.5 + 0.5 * tanh(x/2)
    return FIXED_POINT_SCALE / 2 + tanh_fixed(x / 2) / 2;
}

int32_t ArduinoTinyMLModel::relu_fixed(int32_t x) {
    return (x > 0) ? x : 0;
}

// Matrix operations
void ArduinoTinyMLModel::matrix_init(Matrix* matrix, uint8_t rows, uint8_t cols) {
    matrix->rows = rows;
    matrix->cols = cols;
    memset(matrix->data, 0, sizeof(matrix->data));
}

// Signal processing functions
float ArduinoTinyMLModel::apply_moving_average_filter(float sample) {
    // Add sample to circular buffer
    signal_processor.filter_buffer[signal_processor.filter_index] = (int32_t)(sample * FIXED_POINT_SCALE);
    signal_processor.filter_index = (signal_processor.filter_index + 1) % FILTER_BUFFER_SIZE;
    
    // Apply moving average
    int64_t sum = 0;
    for (int i = 0; i < 5 && i < FILTER_BUFFER_SIZE; i++) {
        int idx = (signal_processor.filter_index - 1 - i + FILTER_BUFFER_SIZE) % FILTER_BUFFER_SIZE;
        sum += (int64_t)signal_processor.filter_buffer[idx] * (signal_processor.filter_coeffs[i] * FIXED_POINT_SCALE);
    }
    
    return (float)(sum / FIXED_POINT_SCALE) / FIXED_POINT_SCALE;
}

bool ArduinoTinyMLModel::detect_peak(float sample) {
    // Add sample to peak detection buffer
    signal_processor.peak_buffer[signal_processor.peak_index] = sample;
    signal_processor.peak_index = (signal_processor.peak_index + 1) % PEAK_DETECTION_BUFFER;
    
    // Simple peak detection - check if current sample is local maximum
    if (signal_processor.peak_index >= 3) {
        int curr_idx = (signal_processor.peak_index - 1 + PEAK_DETECTION_BUFFER) % PEAK_DETECTION_BUFFER;
        int prev_idx = (signal_processor.peak_index - 2 + PEAK_DETECTION_BUFFER) % PEAK_DETECTION_BUFFER;
        int prev2_idx = (signal_processor.peak_index - 3 + PEAK_DETECTION_BUFFER) % PEAK_DETECTION_BUFFER;
        
        float current = signal_processor.peak_buffer[curr_idx];
        float prev = signal_processor.peak_buffer[prev_idx];
        float prev2 = signal_processor.peak_buffer[prev2_idx];
        
        // Calculate dynamic threshold
        float mean = 0;
        for (int i = 0; i < PEAK_DETECTION_BUFFER; i++) {
            mean += signal_processor.peak_buffer[i];
        }
        mean /= PEAK_DETECTION_BUFFER;
        
        float threshold = mean + 0.3; // Adjust threshold as needed
        
        if (current > prev && prev > prev2 && current > threshold) {
            uint32_t current_time = millis();
            if (current_time - signal_processor.last_peak_time > 300) { // Minimum 300ms between peaks (200 BPM max)
                signal_processor.last_peak_time = current_time;
                signal_processor.peak_count++;
                return true;
            }
        }
    }
    
    return false;
}

float ArduinoTinyMLModel::estimate_heart_rate() {
    if (signal_processor.peak_count < 2) {
        return signal_processor.heart_rate_estimate;
    }
    
    // Simple heart rate estimation based on peak intervals
    uint32_t current_time = millis();
    static uint32_t first_peak_time = 0;
    
    if (signal_processor.peak_count == 2) {
        first_peak_time = current_time;
    }
    
    if (signal_processor.peak_count > 2) {
        uint32_t time_elapsed = current_time - first_peak_time;
        if (time_elapsed > 0) {
            float heart_rate = (60000.0 * (signal_processor.peak_count - 1)) / time_elapsed;
            // Apply low-pass filter to smooth heart rate estimate
            signal_processor.heart_rate_estimate = 0.8 * signal_processor.heart_rate_estimate + 0.2 * heart_rate;
        }
    }
    
    return signal_processor.heart_rate_estimate;
}

// Preprocessing and postprocessing
int32_t ArduinoTinyMLModel::preprocess_ecg_sample(float raw_ecg) {
    // ECG input is in range 240-500, need to scale to ~0-1 range for model
    // First, normalize to 0-1 range from typical ECG data range
    float normalized_ecg = (raw_ecg - 240.0) / 260.0;  // Map [240,500] to [0,1]
    
    // Clamp to reasonable bounds
    normalized_ecg = constrain(normalized_ecg, 0.0, 1.0);
    
    // Convert to fixed point
    int32_t ecg_fixed = (int32_t)(normalized_ecg * FIXED_POINT_SCALE);
    
    // Apply normalization if calibration is done
    if (model.input_std > FIXED_POINT_SCALE/10) {  // Check if calibration was done
        int64_t calibrated = ((int64_t)(ecg_fixed - model.input_mean) * FIXED_POINT_SCALE) / model.input_std;
        return constrain_int32((int32_t)calibrated, MIN_INT32, MAX_INT32);
    } else {
        // Default normalization - center around 0 and scale
        int32_t centered = ecg_fixed - (FIXED_POINT_SCALE / 2);  // Center around 0
        return constrain_int32(centered, MIN_INT32, MAX_INT32);
    }
}

float ArduinoTinyMLModel::postprocess_ppg_sample(int32_t raw_ppg) {
    // Debug: print raw PPG value occasionally
    static uint32_t debug_counter = 0;
    if (debug_counter % 360 == 0) {  // Print once per second at 360 Hz
        #if ENABLE_SERIAL_DEBUG
        Serial.print("Debug - raw_ppg: ");
        Serial.print(raw_ppg);
        #endif
    }
    debug_counter++;
    
    // Apply tanh activation to bound output
    int32_t bounded_ppg = tanh_fixed(raw_ppg);
    
    // Debug: print bounded value
    if (debug_counter % 360 == 1) {
        #if ENABLE_SERIAL_DEBUG
        Serial.print(", bounded: ");
        Serial.println(bounded_ppg);
        #endif
    }
    
    // Convert back to float and scale to reasonable PPG range
    float ppg_output = (float)bounded_ppg / FIXED_POINT_SCALE;
    
    // Scale and shift to typical PPG signal range (0.1 to 1.5)
    ppg_output = (ppg_output + 1.0) * 0.7 + 0.1;  // Map [-1,1] to [0.1, 1.5]
    
    return ppg_output;
}

// Hardware interface functions
float ArduinoTinyMLModel::read_ecg_from_adc(uint8_t pin) {
    int adc_value = analogRead(pin);
    
    // Convert ADC reading to voltage
    float voltage = (float)adc_value * ECG_REFERENCE_VOLTAGE / ECG_ADC_RESOLUTION;
    
    // Convert voltage to ECG signal (assuming proper amplification/conditioning)
    // This would depend on your specific ECG front-end circuit
    float ecg_signal = (voltage - ECG_REFERENCE_VOLTAGE / 2) * 2; // Center around 0, scale to ±3.3V range
    
    return ecg_signal;
}

void ArduinoTinyMLModel::output_ppg_to_pwm(float ppg_value, uint8_t pin) {
    // Map PPG value (-1 to 1) to PWM range (0 to 255)
    int pwm_value = (int)map_float(ppg_value, -1.0, 1.0, 0, 255);
    pwm_value = constrain(pwm_value, 0, 255);
    
    analogWrite(pin, pwm_value);
}

// Batch processing
void ArduinoTinyMLModel::predict_ppg_batch(const float* ecg_samples, float* ppg_samples, 
                                           uint16_t num_samples) {
    for (uint16_t i = 0; i < num_samples; i++) {
        ppg_samples[i] = predict_ppg_sample(ecg_samples[i]);
    }
}

// Configuration functions
void ArduinoTinyMLModel::set_sampling_rate(float rate) {
    signal_processor.sampling_rate = rate;
}

void ArduinoTinyMLModel::calibrate_input(const float* calibration_data, uint16_t num_samples) {
    if (num_samples == 0) return;
    
    // First normalize the calibration data to the same range as preprocessing
    float normalized_data[num_samples];
    for (uint16_t i = 0; i < num_samples; i++) {
        // Apply same normalization as preprocessing
        float normalized = (calibration_data[i] - 240.0) / 260.0;
        normalized_data[i] = constrain(normalized, 0.0, 1.0);
    }
    
    // Calculate mean of normalized data
    float sum = 0;
    for (uint16_t i = 0; i < num_samples; i++) {
        sum += normalized_data[i];
    }
    float mean = sum / num_samples;
    
    // Calculate standard deviation of normalized data
    float variance_sum = 0;
    for (uint16_t i = 0; i < num_samples; i++) {
        float diff = normalized_data[i] - mean;
        variance_sum += diff * diff;
    }
    float std = sqrt(variance_sum / num_samples);
    
    // Ensure std is not too small to avoid division issues
    if (std < 0.01) std = 0.1;
    
    // Update preprocessing parameters
    model.input_mean = (int32_t)(mean * FIXED_POINT_SCALE);
    model.input_std = (int32_t)(std * FIXED_POINT_SCALE);
    
    #if ENABLE_SERIAL_DEBUG
    Serial.print("Calibrated (normalized) - Mean: ");
    Serial.print(mean, 4);
    Serial.print(", Std: ");
    Serial.println(std, 4);
    Serial.print("Raw ECG range: ");
    Serial.print(calibration_data[0]);
    Serial.print(" - ");
    Serial.println(calibration_data[num_samples-1]);
    #endif
}

void ArduinoTinyMLModel::set_preprocessing_parameters(float mean, float std) {
    model.input_mean = (int32_t)(mean * FIXED_POINT_SCALE);
    model.input_std = (int32_t)(std * FIXED_POINT_SCALE);
}

// Reset functions
void ArduinoTinyMLModel::reset_model() {
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        reset_lstm_state(&model.lstm_layers[layer]);
    }
    memset(model.sequence_buffer, 0, sizeof(model.sequence_buffer));
    model.buffer_index = 0;
}

void ArduinoTinyMLModel::reset_lstm_state(LSTMCell* cell) {
    memset(cell->state.h, 0, sizeof(cell->state.h));
    memset(cell->state.c, 0, sizeof(cell->state.c));
}

void ArduinoTinyMLModel::reset_signal_processor() {
    memset(&signal_processor, 0, sizeof(SignalProcessor));
    signal_processor.sampling_rate = DEFAULT_SAMPLING_RATE;
    signal_processor.heart_rate_estimate = 60.0;
}

// Performance monitoring
void ArduinoTinyMLModel::update_performance_metrics(uint32_t inference_time_us) {
    #if ENABLE_PERFORMANCE_MONITORING
    model.inference_count++;
    model.total_inference_time_us += inference_time_us;
    
    if (inference_time_us > model.max_inference_time_us) {
        model.max_inference_time_us = inference_time_us;
    }
    
    if (inference_time_us < model.min_inference_time_us) {
        model.min_inference_time_us = inference_time_us;
    }
    #endif
}

void ArduinoTinyMLModel::print_performance_metrics() {
    #if ENABLE_SERIAL_DEBUG
    Serial.println("=== Performance Metrics ===");
    Serial.print("Total inferences: ");
    Serial.println(model.inference_count);
    
    if (model.inference_count > 0) {
        uint32_t avg_time = model.total_inference_time_us / model.inference_count;
        Serial.print("Average inference time: ");
        Serial.print(avg_time);
        Serial.println(" μs");
        
        Serial.print("Max inference time: ");
        Serial.print(model.max_inference_time_us);
        Serial.println(" μs");
        
        Serial.print("Min inference time: ");
        Serial.print(model.min_inference_time_us);
        Serial.println(" μs");
        
        float inferences_per_second = 1000000.0 / avg_time;
        Serial.print("Max sample rate: ~");
        Serial.print(inferences_per_second);
        Serial.println(" Hz");
    }
    
    Serial.print("Current heart rate estimate: ");
    Serial.print(signal_processor.heart_rate_estimate);
    Serial.println(" BPM");
    
    Serial.print("Peak count: ");
    Serial.println(signal_processor.peak_count);
    #endif
}

void ArduinoTinyMLModel::print_model_info() {
    #if ENABLE_SERIAL_DEBUG
    Serial.println("=== Model Information ===");
    Serial.print("Input size: ");
    Serial.println(INPUT_SIZE);
    Serial.print("Hidden size: ");
    Serial.println(HIDDEN_SIZE);
    Serial.print("Number of layers: ");
    Serial.println(NUM_LAYERS);
    Serial.print("Sequence length: ");
    Serial.println(SEQUENCE_LENGTH);
    Serial.print("Memory usage: ");
    Serial.print(get_memory_usage());
    Serial.println(" bytes");
    #endif
}

// Utility functions
size_t ArduinoTinyMLModel::get_memory_usage() {
    return sizeof(TinyMLModel) + sizeof(SignalProcessor);
}

float ArduinoTinyMLModel::get_heart_rate_estimate() {
    return signal_processor.heart_rate_estimate;
}

uint32_t ArduinoTinyMLModel::get_peak_count() {
    return signal_processor.peak_count;
}

// Self-test function
bool ArduinoTinyMLModel::self_test() {
    #if ENABLE_SERIAL_DEBUG
    Serial.println("Running self-test...");
    #endif
    
    // Test with synthetic ECG data
    float test_ecg = 0.5;
    float test_ppg = predict_ppg_sample(test_ecg);
    
    // Check if output is reasonable
    bool test_passed = (test_ppg >= -2.0 && test_ppg <= 2.0);
    
    #if ENABLE_SERIAL_DEBUG
    Serial.print("Test ECG: ");
    Serial.println(test_ecg);
    Serial.print("Test PPG: ");
    Serial.println(test_ppg);
    Serial.print("Self-test: ");
    Serial.println(test_passed ? "PASSED" : "FAILED");
    #endif
    
    return test_passed;
}

// Data logging
void ArduinoTinyMLModel::log_data_to_serial(float ecg, float ppg) {
    #if ENABLE_SERIAL_DEBUG
    Serial.print(millis());
    Serial.print(",");
    Serial.print(ecg, 4);
    Serial.print(",");
    Serial.print(ppg, 4);
    Serial.print(",");
    Serial.println(signal_processor.heart_rate_estimate, 1);
    #endif
}

// Global utility functions
float map_float(float x, float in_min, float in_max, float out_min, float out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

int32_t constrain_int32(int32_t value, int32_t min_val, int32_t max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}


uint32_t get_free_memory() {
    // ARM Cortex-M4 memory estimation
    // Note: This is a simplified estimation for ARM-based boards
    // Actual free memory calculation is more complex and depends on the specific board
    
    #ifdef ARDUINO_ARCH_MBED_NANO
        // For Nano 33 BLE Sense (256KB SRAM)
        // This is a rough estimation - actual implementation would require
        // accessing mbed memory management functions
        static uint32_t estimated_free = 200000; // Estimate 200KB free
        return estimated_free;
    #else
        // Fallback for other ARM boards
        return 100000; // Conservative estimate
    #endif
}
