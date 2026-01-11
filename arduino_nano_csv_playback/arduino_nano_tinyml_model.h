/*
Arduino Nano 33 BLE Sense TinyML ECG-to-PPG Model Header
=======================================================

Lightweight LSTM model header for Arduino Nano 33 BLE Sense.
Optimized for ARM Cortex-M4 processor with 256KB SRAM and 1MB Flash.

Features:
- Fixed-point arithmetic for efficiency
- Memory-optimized LSTM implementation  
- Compatible with Arduino IDE and PlatformIO
- Real-time ECG-to-PPG conversion
- Built-in sensor integration support

Hardware Requirements:
- Arduino Nano 33 BLE Sense
- Analog input for ECG (A0-A7)
- Optional: I2S for audio output or SPI for external DAC
- Built-in sensors: IMU, microphone, gesture, proximity, color, pressure
*/

#ifndef ARDUINO_NANO_TINYML_MODEL_H
#define ARDUINO_NANO_TINYML_MODEL_H

#include <Arduino.h>
#include <math.h>

// Model configuration constants
#define INPUT_SIZE 1
#define HIDDEN_SIZE 16
#define NUM_LAYERS 1
#define SEQUENCE_LENGTH 16
#define OUTPUT_SIZE 1

// Fixed-point arithmetic constants
#define FIXED_POINT_SCALE 1000
#define MAX_INT32 2147483647
#define MIN_INT32 -2147483648

// Memory optimization constants
#define MAX_SEQUENCE_BUFFER 32
#define FILTER_BUFFER_SIZE 8
#define PEAK_DETECTION_BUFFER 16

// Sampling and timing constants
#define DEFAULT_SAMPLING_RATE 360  // Hz
#define ECG_ADC_RESOLUTION 1023    // 10-bit ADC
#define ECG_REFERENCE_VOLTAGE 3.3  // V

// Performance monitoring
#define ENABLE_PERFORMANCE_MONITORING 1
#define ENABLE_SERIAL_DEBUG 1
#define SERIAL_BAUD_RATE 115200

// Structure definitions
typedef struct {
    int32_t data[HIDDEN_SIZE * (INPUT_SIZE + HIDDEN_SIZE)];
    uint8_t rows;
    uint8_t cols;
} Matrix;

typedef struct {
    int32_t h[HIDDEN_SIZE];  // Hidden state
    int32_t c[HIDDEN_SIZE];  // Cell state
} LSTMState;

typedef struct {
    Matrix Wf, Wi, Wo, Wg;  // Weight matrices for gates
    int32_t bf[HIDDEN_SIZE]; // Forget gate bias
    int32_t bi[HIDDEN_SIZE]; // Input gate bias
    int32_t bo[HIDDEN_SIZE]; // Output gate bias
    int32_t bg[HIDDEN_SIZE]; // Candidate gate bias
    LSTMState state;
    uint8_t input_size;
    uint8_t hidden_size;
} LSTMCell;

typedef struct {
    LSTMCell lstm_layers[NUM_LAYERS];
    int32_t output_weights[HIDDEN_SIZE];
    int32_t output_bias;
    
    // Preprocessing parameters
    int32_t input_mean;
    int32_t input_std;
    
    // Sequence processing
    int32_t sequence_buffer[SEQUENCE_LENGTH];
    uint8_t buffer_index;
    
    // Performance metrics
    uint32_t inference_count;
    uint32_t total_inference_time_us;
    uint32_t max_inference_time_us;
    uint32_t min_inference_time_us;
} TinyMLModel;

typedef struct {
    float ecg_buffer[MAX_SEQUENCE_BUFFER];
    float ppg_buffer[MAX_SEQUENCE_BUFFER];
    int32_t filter_buffer[FILTER_BUFFER_SIZE];
    float peak_buffer[PEAK_DETECTION_BUFFER];
    uint8_t ecg_index;
    uint8_t ppg_index;
    uint8_t filter_index;
    uint8_t peak_index;
    
    // Signal processing parameters
    float sampling_rate;
    float heart_rate_estimate;
    uint32_t last_peak_time;
    uint32_t peak_count;
    
    // Filter coefficients (moving average)
    float filter_coeffs[5] = {0.1, 0.2, 0.4, 0.2, 0.1};
} SignalProcessor;

// Function declarations
class ArduinoTinyMLModel {
private:
    TinyMLModel model;
    SignalProcessor signal_processor;
    
    // Matrix operations
    void matrix_multiply(const Matrix* a, const Matrix* b, Matrix* result);
    void matrix_add(const Matrix* a, const Matrix* b, Matrix* result);
    void matrix_init(Matrix* matrix, uint8_t rows, uint8_t cols);
    
    // Activation functions (made static for function pointers)
    static int32_t tanh_fixed(int32_t x);
    static int32_t sigmoid_fixed(int32_t x);
    static int32_t relu_fixed(int32_t x);
    
    // LSTM operations
    void compute_gate(const int32_t* input, const Matrix* weights, 
                     const int32_t* bias, int32_t* output, 
                     int32_t (*activation)(int32_t));
    void lstm_forward(LSTMCell* cell, const int32_t* input, int32_t* output);
    void reset_lstm_state(LSTMCell* cell);
    
    // Signal processing
    float apply_moving_average_filter(float sample);
    bool detect_peak(float sample);
    float estimate_heart_rate();
    
    // Preprocessing and postprocessing
    int32_t preprocess_ecg_sample(float raw_ecg);
    float postprocess_ppg_sample(int32_t raw_ppg);
    
    // Memory and performance monitoring
    void update_performance_metrics(uint32_t inference_time_us);
    size_t get_memory_usage();

public:
    ArduinoTinyMLModel();
    ~ArduinoTinyMLModel();
    
    // Core functions
    bool initialize();
    bool load_model_weights();
    float predict_ppg_sample(float ecg_sample);
    void reset_model();
    
    // Batch processing
    void predict_ppg_batch(const float* ecg_samples, float* ppg_samples, 
                          uint16_t num_samples);
    
    // Configuration and calibration
    void set_sampling_rate(float rate);
    void calibrate_input(const float* calibration_data, uint16_t num_samples);
    void set_preprocessing_parameters(float mean, float std);
    
    // Performance and diagnostics
    void print_performance_metrics();
    void print_model_info();
    bool self_test();
    
    // Signal processing utilities
    float get_heart_rate_estimate();
    uint32_t get_peak_count();
    void reset_signal_processor();
    
    // Hardware interface helpers
    float read_ecg_from_adc(uint8_t pin);
    void output_ppg_to_dac(float ppg_value);
    void output_ppg_to_pwm(float ppg_value, uint8_t pin);
    
    // Data logging and visualization
    void log_data_to_serial(float ecg, float ppg);
    void send_data_bluetooth(float ecg, float ppg);
};

// Global utility functions
float map_float(float x, float in_min, float in_max, float out_min, float out_max);
int32_t constrain_int32(int32_t value, int32_t min_val, int32_t max_val);
uint32_t get_free_memory();

// Model weight declarations (will be defined in weights file)
extern const int32_t model_weights_wf[HIDDEN_SIZE][INPUT_SIZE + HIDDEN_SIZE];
extern const int32_t model_weights_wi[HIDDEN_SIZE][INPUT_SIZE + HIDDEN_SIZE];
extern const int32_t model_weights_wo[HIDDEN_SIZE][INPUT_SIZE + HIDDEN_SIZE];
extern const int32_t model_weights_wg[HIDDEN_SIZE][INPUT_SIZE + HIDDEN_SIZE];
extern const int32_t model_bias_f[HIDDEN_SIZE];
extern const int32_t model_bias_i[HIDDEN_SIZE];
extern const int32_t model_bias_o[HIDDEN_SIZE];
extern const int32_t model_bias_g[HIDDEN_SIZE];
extern const int32_t output_layer_weights[HIDDEN_SIZE];
extern const int32_t output_layer_bias;

#endif // ARDUINO_NANO_TINYML_MODEL_H
