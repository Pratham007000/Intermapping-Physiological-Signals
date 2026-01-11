import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleAttention(nn.Module):
    """Simplified attention mechanism"""
    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_size]
        attention_scores = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
        context_vector = torch.sum(lstm_output * attention_weights, dim=1)  # [batch_size, hidden_size]
        return context_vector, attention_weights

class SimplifiedECGtoSCG(nn.Module):
    """
    Simplified ECG to SCG model optimized for CPU training.
    Uses proven techniques without complex operations that cause tensor size issues.
    """
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1, 
                 dropout=0.3, use_attention=True):
        super(SimplifiedECGtoSCG, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # 1D Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(input_size, hidden_size//4, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(hidden_size//4, hidden_size//2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_size//2, hidden_size//2, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(hidden_size//4)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        self.bn3 = nn.BatchNorm1d(hidden_size//2)
        
        self.dropout_conv = nn.Dropout(dropout * 0.5)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size//2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        lstm_output_size = hidden_size * 2  # Bidirectional
        
        # Attention mechanism
        if use_attention:
            self.attention = SimpleAttention(lstm_output_size)
            self.use_sequence_output = False
        else:
            self.use_sequence_output = True
        
        # Batch normalization after LSTM
        self.bn_lstm = nn.BatchNorm1d(lstm_output_size)
        
        # Output layers
        if use_attention:
            # For attention: use context vector
            self.fc_layers = nn.Sequential(
                nn.Linear(lstm_output_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_size//2, output_size)
            )
        else:
            # For sequence output: direct mapping
            self.fc_layers = nn.Sequential(
                nn.Linear(lstm_output_size, hidden_size//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size//2, output_size)
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier/He initialization"""
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
        
        # Transpose for conv1d: (batch, channels, length)
        x_conv = x.transpose(1, 2)
        
        # Convolutional feature extraction
        x_conv = F.relu(self.bn1(self.conv1(x_conv)))
        x_conv = self.dropout_conv(x_conv)
        
        x_conv = F.relu(self.bn2(self.conv2(x_conv)))
        x_conv = self.dropout_conv(x_conv)
        
        x_conv = F.relu(self.bn3(self.conv3(x_conv)))
        x_conv = self.dropout_conv(x_conv)
        
        # Transpose back for LSTM: (batch, length, channels)
        x_lstm = x_conv.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_lstm)
        
        if self.use_attention:
            # Use attention to get context vector
            context_vector, attention_weights = self.attention(lstm_out)
            
            # Expand context vector to sequence length
            output = context_vector.unsqueeze(1).repeat(1, seq_len, 1)
            
            # Apply batch norm (reshape for batch norm)
            output_reshaped = output.contiguous().view(-1, output.size(2))
            output_normalized = self.bn_lstm(output_reshaped)
            output = output_normalized.view(batch_size, seq_len, -1)
            
            # Apply fully connected layers
            output = self.fc_layers(output)
            
        else:
            # Use sequence output directly
            # Apply batch norm
            lstm_out_reshaped = lstm_out.contiguous().view(-1, lstm_out.size(2))
            lstm_out_normalized = self.bn_lstm(lstm_out_reshaped)
            lstm_out = lstm_out_normalized.view(batch_size, seq_len, -1)
            
            # Apply fully connected layers
            output = self.fc_layers(lstm_out)
        
        return output

class SimplifiedLoss(nn.Module):
    """Simplified loss function for CPU training"""
    def __init__(self, mse_weight=1.0, mae_weight=0.3, corr_weight=1.5, smooth_weight=0.1):
        super(SimplifiedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.corr_weight = corr_weight
        self.smooth_weight = smooth_weight
        
    def correlation_loss(self, y_true, y_pred):
        """Negative Pearson correlation loss"""
        # Flatten the tensors
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        
        # Center the data
        y_true_centered = y_true_flat - torch.mean(y_true_flat)
        y_pred_centered = y_pred_flat - torch.mean(y_pred_flat)
        
        # Calculate correlation
        numerator = torch.sum(y_true_centered * y_pred_centered)
        denominator = torch.sqrt(torch.sum(y_true_centered**2) * 
                               torch.sum(y_pred_centered**2) + 1e-8)
        
        correlation = numerator / denominator
        return -correlation
    
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
        smooth_loss = self.smoothness_loss(y_pred)
        
        # Combine losses
        total_loss = (self.mse_weight * mse_loss + 
                     self.mae_weight * mae_loss +
                     self.corr_weight * corr_loss +
                     self.smooth_weight * smooth_loss)
        
        return total_loss

