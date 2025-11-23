import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), pool_size=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_size) if pool_size else None
        
    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        if self.pool:
            x = self.pool(x)
        return x

class SELD_CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) for SELD.
    
    Architecture:
    1. CNN Encoder: Extracts features from log-mel spectrograms, reducing frequency dimension.
    2. RNN (GRU): Models temporal dependencies.
    3. Fully Connected Head: Maps features to the output grid (I, J, M).
    """
    def __init__(self, n_channels=4, n_mels=64, grid_size=(18, 36), num_classes=14, 
                 cnn_channels=[64, 128, 256, 512], rnn_hidden=256, rnn_layers=2, dropout=0.3):
        super().__init__()
        self.I, self.J = grid_size
        self.grid_cells = self.I * self.J
        self.num_classes = num_classes
        
        # --- CNN Encoder ---
        self.cnn_blocks = nn.ModuleList()
        current_channels = n_channels
        current_freq = n_mels
        
        # We want to reduce frequency dimension to 1 or 2
        # Standard SELDnet-like pooling
        pool_sizes = [
            (1, 2), # Freq / 2
            (1, 2), # Freq / 4
            (1, 2), # Freq / 8
            (1, 2)  # Freq / 16 -> 64/16 = 4
        ]
        
        for i, out_channels in enumerate(cnn_channels):
            pool = pool_sizes[i] if i < len(pool_sizes) else None
            self.cnn_blocks.append(
                ConvBlock(current_channels, out_channels, pool_size=pool)
            )
            current_channels = out_channels
            if pool:
                current_freq = current_freq // pool[1]
        
        self.cnn_out_channels = current_channels
        self.cnn_out_freq = current_freq
        
        # Projection to RNN input size
        self.rnn_input_size = self.cnn_out_channels * self.cnn_out_freq
        
        # --- RNN (Temporal Modeling) ---
        self.rnn = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0
        )
        
        self.rnn_out_size = rnn_hidden * 2 # Bidirectional
        
        # --- Fully Connected Head ---
        self.fnn = nn.Sequential(
            nn.Linear(self.rnn_out_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.grid_cells * num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C, F) - Batch, Time, Channels, Freq (Mel bins)
        Returns:
            logits: (B, T, G, M)
        """
        B, T, C, F = x.shape
        
        # Permute for CNN: (B, T, C, F) -> (B, C, T, F)
        # We treat Time as the "Width" and Freq as "Height" of the image
        # But standard Conv2d expects (N, C, H, W). 
        # Here we can treat the entire (T, F) as an image if we want to pool over time too.
        # BUT, for SELD, we usually want to preserve Time resolution or reduce it by a fixed factor.
        # The current dataset provides windows of 5s.
        
        # Approach: Treat T as the "H" dimension and F as "W" dimension? 
        # Or simply reshape (B*T, C, F, 1) like the previous model?
        # NO, CRNN usually processes the whole sequence.
        
        # Standard CRNN input: (B, C, T, F)
        x = x.permute(0, 2, 1, 3) # (B, C, T, F)
        
        # Apply CNN blocks
        for block in self.cnn_blocks:
            x = block(x)
            
        # x shape: (B, C_out, T, F_out)
        # We need to merge C_out and F_out for RNN
        x = x.permute(0, 2, 1, 3) # (B, T, C_out, F_out)
        B, T, C_out, F_out = x.shape
        x = x.reshape(B, T, C_out * F_out)
        
        # Apply RNN
        x, _ = self.rnn(x) # (B, T, rnn_out_size)
        
        # Apply Head
        logits = self.fnn(x) # (B, T, G*M)
        
        # Reshape to grid
        logits = logits.view(B, T, self.grid_cells, self.num_classes)
        
        return logits
