import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-like attention"""
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        residual = x
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        return self.layer_norm(output + residual)

class ConvBlock(nn.Module):
    """1D Convolutional block with residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += residual
        return F.relu(out)

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network with dilated convolutions"""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation
            
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size,
                                   dilation=dilation, padding=padding))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            # Causal padding (remove future information)
            layers.append(nn.ConstantPad1d((0, -padding), 0))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class WaveletTransform(nn.Module):
    """Learnable wavelet-like transformation"""
    def __init__(self, in_channels, out_channels, scales=4):
        super(WaveletTransform, self).__init__()
        self.scales = scales
        self.filters = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // scales, kernel_size=2**i+1, 
                     padding=2**i//2, dilation=1) for i in range(scales)
        ])
        self.combine = nn.Conv1d(out_channels, out_channels, 1)
        
    def forward(self, x):
        outputs = []
        for filter_layer in self.filters:
            outputs.append(filter_layer(x))
        
        combined = torch.cat(outputs, dim=1)
        return self.combine(combined)

class ImprovedECGtoSCG(nn.Module):
    """
    Improved ECG to SCG estimation model with multiple advanced components:
    - Multi-scale feature extraction
    - Self-attention mechanisms
    - Temporal convolutional networks
    - Residual connections
    - Frequency domain processing
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, output_size=1, 
                 dropout=0.2, use_attention=True, use_tcn=True):
        super(ImprovedECGtoSCG, self).__init__()
        
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        self.use_tcn = use_tcn
        
        # Multi-scale feature extraction
        self.wavelet_transform = WaveletTransform(input_size, hidden_size//2)
        
        # Convolutional feature extraction
        self.conv_blocks = nn.ModuleList([
            ConvBlock(input_size, hidden_size//4, kernel_size=7, padding=3),
            ConvBlock(hidden_size//4, hidden_size//2, kernel_size=5, padding=2),
            ConvBlock(hidden_size//2, hidden_size, kernel_size=3, padding=1)
        ])
        
        # Temporal Convolutional Network
        if use_tcn:
            tcn_channels = [hidden_size, hidden_size, hidden_size//2]
            self.tcn = TemporalConvNet(hidden_size, tcn_channels, dropout=dropout)
            lstm_input_size = hidden_size//2
        else:
            lstm_input_size = hidden_size
        
        # Enhanced LSTM with multiple layers
        self.lstm = nn.LSTM(
            input_size=lstm_input_size + hidden_size//2,  # TCN + Wavelet features
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        lstm_output_size = hidden_size * 2  # Bidirectional
        
        # Multi-head self-attention
        if use_attention:
            self.pos_encoding = PositionalEncoding(lstm_output_size)
            self.attention_layers = nn.ModuleList([
                MultiHeadAttention(lstm_output_size, n_heads=8, dropout=dropout)
                for _ in range(2)
            ])
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-scale output heads
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_size, output_size),  # Direct prediction
            nn.Linear(hidden_size, output_size),  # Low-frequency component
            nn.Linear(hidden_size, output_size),  # High-frequency component
        ])
        
        # Output combination
        self.output_combine = nn.Linear(3 * output_size, output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Transpose for conv1d (batch, channels, length)
        x_conv = x.transpose(1, 2)
        
        # Multi-scale wavelet features
        wavelet_features = self.wavelet_transform(x_conv)
        
        # Convolutional feature extraction
        conv_features = x_conv
        for conv_block in self.conv_blocks:
            conv_features = conv_block(conv_features)
        
        # Temporal Convolutional Network
        if self.use_tcn:
            tcn_features = self.tcn(conv_features)
        else:
            tcn_features = conv_features
        
        # Transpose back for LSTM
        tcn_features = tcn_features.transpose(1, 2)
        wavelet_features = wavelet_features.transpose(1, 2)
        
        # Combine features
        combined_features = torch.cat([tcn_features, wavelet_features], dim=-1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined_features)
        
        # Self-attention
        if self.use_attention:
            lstm_out = self.pos_encoding(lstm_out)
            for attention_layer in self.attention_layers:
                lstm_out = attention_layer(lstm_out)
        
        # Feature fusion
        fused_features = self.feature_fusion(lstm_out)
        
        # Multi-scale predictions
        outputs = []
        for head in self.output_heads:
            outputs.append(head(fused_features))
        
        # Combine outputs
        combined_output = torch.cat(outputs, dim=-1)
        final_output = self.output_combine(combined_output)
        
        return final_output

class AdvancedLoss(nn.Module):
    """Advanced loss function combining multiple objectives"""
    def __init__(self, mse_weight=1.0, mae_weight=0.5, corr_weight=2.0, 
                 freq_weight=0.3, smooth_weight=0.1, phase_weight=0.2):
        super(AdvancedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.corr_weight = corr_weight
        self.freq_weight = freq_weight
        self.smooth_weight = smooth_weight
        self.phase_weight = phase_weight
        
    def correlation_loss(self, y_true, y_pred):
        """Negative Pearson correlation loss"""
        y_true_centered = y_true - torch.mean(y_true, dim=1, keepdim=True)
        y_pred_centered = y_pred - torch.mean(y_pred, dim=1, keepdim=True)
        
        numerator = torch.sum(y_true_centered * y_pred_centered, dim=1)
        denominator = torch.sqrt(torch.sum(y_true_centered**2, dim=1) * 
                               torch.sum(y_pred_centered**2, dim=1) + 1e-8)
        
        correlation = numerator / denominator
        return -torch.mean(correlation)
    
    def frequency_domain_loss(self, y_true, y_pred):
        """Loss in frequency domain using FFT"""
        # Take FFT
        y_true_fft = torch.fft.fft(y_true.squeeze(-1))
        y_pred_fft = torch.fft.fft(y_pred.squeeze(-1))
        
        # Magnitude loss
        mag_loss = F.mse_loss(torch.abs(y_true_fft), torch.abs(y_pred_fft))
        
        return mag_loss
    
    def phase_loss(self, y_true, y_pred):
        """Phase relationship loss"""
        y_true_fft = torch.fft.fft(y_true.squeeze(-1))
        y_pred_fft = torch.fft.fft(y_pred.squeeze(-1))
        
        # Phase difference
        phase_true = torch.angle(y_true_fft)
        phase_pred = torch.angle(y_pred_fft)
        
        # Circular loss for phase
        phase_diff = torch.sin(phase_true - phase_pred)
        return torch.mean(phase_diff**2)
    
    def smoothness_loss(self, y_pred):
        """Smoothness penalty"""
        diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        return torch.mean(diff**2)
    
    def forward(self, y_pred, y_true):
        # Basic losses
        mse_loss = F.mse_loss(y_pred, y_true)
        mae_loss = F.l1_loss(y_pred, y_true)
        
        # Advanced losses
        corr_loss = self.correlation_loss(y_true, y_pred)
        freq_loss = self.frequency_domain_loss(y_true, y_pred)
        phase_loss = self.phase_loss(y_true, y_pred)
        smooth_loss = self.smoothness_loss(y_pred)
        
        # Combine losses
        total_loss = (self.mse_weight * mse_loss + 
                     self.mae_weight * mae_loss +
                     self.corr_weight * corr_loss +
                     self.freq_weight * freq_loss +
                     self.phase_weight * phase_loss +
                     self.smooth_weight * smooth_loss)
        
        return total_loss

