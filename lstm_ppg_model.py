import torch
import torch.nn as nn

class ECGtoPPG_LSTM(nn.Module):
    """
    Enhanced LSTM model for ECG to PPG signal conversion.
    
    This model uses a stacked LSTM architecture with dropout and batch normalization
    to improve training stability and model performance. The bidirectional option
    allows the model to capture temporal dependencies in both directions.
    
    Parameters:
    -----------
    input_size : int, default=1
        Number of expected features in the input (typically 1 for ECG signal)
    hidden_size : int, default=128
        Number of features in the hidden state of the LSTM
    num_layers : int, default=2
        Number of recurrent layers (stacked LSTM)
    output_size : int, default=1
        Number of expected features in the output (typically 1 for PPG signal)
    bidirectional : bool, default=True
        If True, becomes a bidirectional LSTM
    dropout_rate : float, default=0.3
        Dropout probability for regularization between LSTM layers (0-1)
    """
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1, 
                 bidirectional=True, dropout_rate=0.3):
        super(ECGtoPPG_LSTM, self).__init__()
        
        # Store model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        
        # Main LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0  # Dropout between LSTM layers
        )
        
        # Determine output features from LSTM based on bidirectionality
        lstm_output_features = hidden_size * (2 if bidirectional else 1)
        
        # Batch normalization for LSTM outputs
        self.batch_norm = nn.BatchNorm1d(lstm_output_features)
        
        # Fully connected output layer
        self.fc = nn.Linear(lstm_output_features, output_size)

    def forward(self, x):
        """
        Forward pass of the LSTM model.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size * num_directions)
        
        # Reshape for batch normalization (which expects [N, C, ...])
        # From (batch_size, seq_len, features) to (batch_size * seq_len, features)
        reshaped = lstm_out.contiguous().view(-1, lstm_out.size(2))
        
        # Apply batch normalization
        normalized = self.batch_norm(reshaped)
        
        # Reshape back to (batch_size, seq_len, features)
        normalized = normalized.view(batch_size, seq_len, -1)
        
        # Apply fully connected layer
        out = self.fc(normalized)  # Shape: (batch_size, seq_len, output_size)
        
        return out
