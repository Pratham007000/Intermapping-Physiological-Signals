"""
PyTorch to Pico Weight Converter
===============================

Converts trained PyTorch TinyML model weights to Raspberry Pi Pico compatible format.
Handles quantization and fixed-point conversion for embedded deployment.

Usage:
python convert_weights_to_pico.py

Output:
- pico_model_weights.py: MicroPython file with model weights
- pico_model_weights.json: JSON file with weights (optional)
"""

import torch
import numpy as np
import json
import os
from tinyml_lstm_model import TinyECGtoPPG_LSTM

# Fixed-point scale factor (must match pico_tinyml_model.py)
FIXED_POINT_SCALE = 1000

def load_pytorch_model(model_path="best_tinyml_ppg_model.pth"):
    """Load the trained PyTorch model."""
    print(f"Loading PyTorch model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return None
    
    # Create model with same architecture as training
    model = TinyECGtoPPG_LSTM(
        input_size=1,
        hidden_size=48,
        num_layers=2,
        output_size=1,
        dropout_rate=0.1
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    model.eval()
    
    print("PyTorch model loaded successfully!")
    return model

def extract_lstm_weights(lstm_layer, layer_idx=0):
    """Extract LSTM weights from PyTorch layer."""
    print(f"Extracting LSTM layer {layer_idx} weights...")
    
    # PyTorch LSTM weights are organized as:
    # weight_ih_l[k] : input-to-hidden weights for layer k
    # weight_hh_l[k] : hidden-to-hidden weights for layer k  
    # bias_ih_l[k] : input-to-hidden bias for layer k
    # bias_hh_l[k] : hidden-to-hidden bias for layer k
    
    # Each contains 4 gates in order: input, forget, cell, output (i, f, g, o)
    
    weight_ih = lstm_layer.weight_ih_l0.data.numpy() if layer_idx == 0 else lstm_layer.__getattr__(f'weight_ih_l{layer_idx}').data.numpy()
    weight_hh = lstm_layer.weight_hh_l0.data.numpy() if layer_idx == 0 else lstm_layer.__getattr__(f'weight_hh_l{layer_idx}').data.numpy()
    bias_ih = lstm_layer.bias_ih_l0.data.numpy() if layer_idx == 0 else lstm_layer.__getattr__(f'bias_ih_l{layer_idx}').data.numpy()
    bias_hh = lstm_layer.bias_hh_l0.data.numpy() if layer_idx == 0 else lstm_layer.__getattr__(f'bias_hh_l{layer_idx}').data.numpy()
    
    hidden_size = weight_hh.shape[1]
    input_size = weight_ih.shape[1]
    
    # Split into gates (PyTorch order: i, f, g, o)
    # We need order: f, i, o, g for our Pico implementation
    
    # Extract gate weights (4 gates * hidden_size each)
    Wi = weight_ih[0*hidden_size:1*hidden_size, :]  # Input gate
    Wf = weight_ih[1*hidden_size:2*hidden_size, :]  # Forget gate  
    Wg = weight_ih[2*hidden_size:3*hidden_size, :]  # Cell gate
    Wo = weight_ih[3*hidden_size:4*hidden_size, :]  # Output gate
    
    Ui = weight_hh[0*hidden_size:1*hidden_size, :]  # Input gate (hidden)
    Uf = weight_hh[1*hidden_size:2*hidden_size, :]  # Forget gate (hidden)
    Ug = weight_hh[2*hidden_size:3*hidden_size, :]  # Cell gate (hidden)
    Uo = weight_hh[3*hidden_size:4*hidden_size, :]  # Output gate (hidden)
    
    # Extract biases
    bi_ih = bias_ih[0*hidden_size:1*hidden_size]
    bf_ih = bias_ih[1*hidden_size:2*hidden_size]
    bg_ih = bias_ih[2*hidden_size:3*hidden_size]
    bo_ih = bias_ih[3*hidden_size:4*hidden_size]
    
    bi_hh = bias_hh[0*hidden_size:1*hidden_size]
    bf_hh = bias_hh[1*hidden_size:2*hidden_size]
    bg_hh = bias_hh[2*hidden_size:3*hidden_size]
    bo_hh = bias_hh[3*hidden_size:4*hidden_size]
    
    # Combine input and hidden weights for each gate
    Wf_combined = np.concatenate([Wf, Uf], axis=1)  # [hidden_size, input_size + hidden_size]
    Wi_combined = np.concatenate([Wi, Ui], axis=1)
    Wo_combined = np.concatenate([Wo, Uo], axis=1)
    Wg_combined = np.concatenate([Wg, Ug], axis=1)
    
    # Combine biases
    bf_combined = bf_ih + bf_hh
    bi_combined = bi_ih + bi_hh
    bo_combined = bo_ih + bo_hh
    bg_combined = bg_ih + bg_hh
    
    return {
        'Wf': Wf_combined,
        'Wi': Wi_combined, 
        'Wo': Wo_combined,
        'Wg': Wg_combined,
        'bf': bf_combined,
        'bi': bi_combined,
        'bo': bo_combined,
        'bg': bg_combined,
        'input_size': input_size,
        'hidden_size': hidden_size
    }

def quantize_weights(weights_dict, scale=FIXED_POINT_SCALE):
    """Convert floating point weights to fixed-point integers."""
    print("Quantizing weights to fixed-point...")
    
    quantized = {}
    
    for key, value in weights_dict.items():
        if isinstance(value, np.ndarray):
            # Convert to fixed-point integers
            quantized_array = np.round(value * scale).astype(np.int32)
            # Clamp to prevent overflow
            quantized_array = np.clip(quantized_array, -2147483647, 2147483647)
            quantized[key] = quantized_array.tolist()  # Convert to list for JSON serialization
        else:
            quantized[key] = value  # Keep non-array values as-is
    
    return quantized

def create_reduced_model_weights(full_weights, target_hidden_size=16):
    """Create reduced model weights for Pico constraints."""
    print(f"Reducing model size to {target_hidden_size} hidden units...")
    
    reduced_weights = {}
    original_hidden_size = full_weights['hidden_size']
    input_size = full_weights['input_size']
    
    # Reduce hidden size by taking first N units
    reduction_ratio = target_hidden_size / original_hidden_size
    
    for key in ['Wf', 'Wi', 'Wo', 'Wg']:
        original_weight = np.array(full_weights[key])
        # Take first target_hidden_size rows
        reduced_weight = original_weight[:target_hidden_size, :]
        reduced_weights[key] = reduced_weight
    
    for key in ['bf', 'bi', 'bo', 'bg']:
        original_bias = np.array(full_weights[key])
        # Take first target_hidden_size elements
        reduced_bias = original_bias[:target_hidden_size]
        reduced_weights[key] = reduced_bias
    
    reduced_weights['input_size'] = input_size
    reduced_weights['hidden_size'] = target_hidden_size
    
    print(f"Model reduced from {original_hidden_size} to {target_hidden_size} units")
    return reduced_weights

def save_weights_as_micropython(weights_dict, output_path="pico_model_weights.py"):
    """Save weights as MicroPython compatible file."""
    print(f"Saving weights to {output_path}...")
    
    with open(output_path, 'w') as f:
        f.write('"""\n')
        f.write('Raspberry Pi Pico TinyML Model Weights\n')
        f.write('=====================================\n\n')
        f.write('Pre-trained LSTM weights converted from PyTorch model.\n')
        f.write('Fixed-point format for efficient computation on RP2040.\n')
        f.write('"""\n\n')
        
        f.write(f'# Model configuration\n')
        f.write(f'INPUT_SIZE = {weights_dict["input_size"]}\n')
        f.write(f'HIDDEN_SIZE = {weights_dict["hidden_size"]}\n')
        f.write(f'FIXED_POINT_SCALE = {FIXED_POINT_SCALE}\n\n')
        
        # Write weight matrices
        for gate in ['f', 'i', 'o', 'g']:
            weight_key = f'W{gate}'
            bias_key = f'b{gate}'
            
            f.write(f'# {gate.upper()} gate weights and biases\n')
            f.write(f'{weight_key.upper()}_WEIGHTS = [\n')
            
            weight_matrix = weights_dict[weight_key]
            for row in weight_matrix:
                f.write('    [')
                f.write(', '.join(map(str, row)))
                f.write('],\n')
            f.write(']\n\n')
            
            f.write(f'{bias_key.upper()}_BIAS = ')
            f.write(str(weights_dict[bias_key].tolist() if hasattr(weights_dict[bias_key], 'tolist') else weights_dict[bias_key]))
            f.write('\n\n')
        
        # Write helper function
        f.write('def get_model_weights():\n')
        f.write('    """Return dictionary of model weights."""\n')
        f.write('    return {\n')
        f.write('        "Wf": WF_WEIGHTS,\n')
        f.write('        "Wi": WI_WEIGHTS,\n')
        f.write('        "Wo": WO_WEIGHTS,\n')
        f.write('        "Wg": WG_WEIGHTS,\n')
        f.write('        "bf": BF_BIAS,\n')
        f.write('        "bi": BI_BIAS,\n')
        f.write('        "bo": BO_BIAS,\n')
        f.write('        "bg": BG_BIAS,\n')
        f.write('        "input_size": INPUT_SIZE,\n')
        f.write('        "hidden_size": HIDDEN_SIZE\n')
        f.write('    }\n')
    
    print(f"MicroPython weights saved to {output_path}")

def save_weights_as_json(weights_dict, output_path="pico_model_weights.json"):
    """Save weights as JSON file (optional)."""
    print(f"Saving weights to {output_path}...")
    
    # Convert numpy arrays to lists for JSON serialization
    json_weights = {}
    for key, value in weights_dict.items():
        if isinstance(value, np.ndarray):
            json_weights[key] = value.tolist()
        else:
            json_weights[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(json_weights, f, indent=2)
    
    print(f"JSON weights saved to {output_path}")

def calculate_model_size(weights_dict):
    """Calculate approximate model size in bytes."""
    total_parameters = 0
    
    for key in ['Wf', 'Wi', 'Wo', 'Wg']:
        weight_matrix = weights_dict[key]
        if isinstance(weight_matrix, np.ndarray):
            total_parameters += weight_matrix.size
        else:
            total_parameters += len(weight_matrix) * len(weight_matrix[0])
    
    for key in ['bf', 'bi', 'bo', 'bg']:
        bias_vector = weights_dict[key]
        if isinstance(bias_vector, np.ndarray):
            total_parameters += bias_vector.size
        else:
            total_parameters += len(bias_vector)
    
    # 4 bytes per 32-bit integer parameter
    model_size_bytes = total_parameters * 4
    
    return total_parameters, model_size_bytes

def main():
    """Main conversion process."""
    print("PyTorch to Raspberry Pi Pico Weight Converter")
    print("=" * 50)
    
    # Load PyTorch model
    pytorch_model = load_pytorch_model()
    if pytorch_model is None:
        return
    
    # Extract LSTM weights from first layer (we'll use single layer for Pico)
    lstm_weights = extract_lstm_weights(pytorch_model.lstm, layer_idx=0)
    
    # Create reduced model for Pico memory constraints
    reduced_weights = create_reduced_model_weights(lstm_weights, target_hidden_size=16)
    
    # Quantize weights
    quantized_weights = quantize_weights(reduced_weights)
    
    # Calculate model size
    param_count, model_size = calculate_model_size(quantized_weights)
    print(f"\nModel Statistics:")
    print(f"Parameters: {param_count:,}")
    print(f"Model size: {model_size:,} bytes ({model_size/1024:.1f} KB)")
    
    # Check if model fits in Pico memory constraints
    pico_flash_budget = 100 * 1024  # 100KB budget for model
    pico_ram_budget = 50 * 1024     # 50KB RAM budget
    
    if model_size < pico_flash_budget:
        print(f"✅ Model fits in Pico flash memory ({model_size/1024:.1f} KB < {pico_flash_budget/1024:.1f} KB)")
    else:
        print(f"❌ Model too large for Pico flash memory ({model_size/1024:.1f} KB > {pico_flash_budget/1024:.1f} KB)")
    
    # Save weights in MicroPython format
    save_weights_as_micropython(quantized_weights)
    
    # Optionally save as JSON
    save_weights_as_json(quantized_weights)
    
    print("\n✅ Weight conversion completed successfully!")
    print("\nNext steps:")
    print("1. Copy pico_model_weights.py to your Raspberry Pi Pico")
    print("2. Update pico_tinyml_model.py to load the converted weights")
    print("3. Run the main application on Pico")

if __name__ == "__main__":
    main()
