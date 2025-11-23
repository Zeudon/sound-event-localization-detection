import torch
import torch.nn as nn
import torch.nn.functional as F
from model_crnn import ConvBlock

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.swish = Swish()

    def forward(self, x):
        # x: (B, T, D)
        x_res = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x_res + 0.5 * x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        x_res = x
        x = self.norm(x)
        
        q = self.w_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v) # (B, H, T, D_h)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        
        out = self.w_o(out)
        out = self.dropout(out)
        
        return x_res + out

class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=d_model, bias=True)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        x_res = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2) # (B, D, T)
        
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2) # (B, T, D)
        return x_res + x

class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, d_ff=1024, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, d_ff, dropout)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.conv = ConformerConvModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForward(d_model, d_ff, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ff1(x)
        x = self.attn(x)
        x = self.conv(x)
        x = self.ff2(x)
        x = self.norm(x)
        return x

class SELD_Conformer(nn.Module):
    """
    Conformer-based SELD model.
    
    Architecture:
    1. CNN Encoder: Extracts features from log-mel spectrograms.
    2. Conformer Encoder: Models temporal dependencies with Attention + CNN.
    3. Fully Connected Head: Maps features to the output grid.
    """
    def __init__(self, n_channels=4, n_mels=64, grid_size=(18, 36), num_classes=14, 
                 cnn_channels=[64, 128, 256, 512], 
                 conf_d_model=256, conf_n_heads=4, conf_n_layers=2, conf_kernel_size=31, dropout=0.3):
        super().__init__()
        self.I, self.J = grid_size
        self.grid_cells = self.I * self.J
        self.num_classes = num_classes
        
        # --- CNN Encoder (Same as CRNN) ---
        self.cnn_blocks = nn.ModuleList()
        current_channels = n_channels
        current_freq = n_mels
        
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
        
        # Projection to Conformer input size
        self.cnn_feat_size = self.cnn_out_channels * self.cnn_out_freq
        self.proj = nn.Linear(self.cnn_feat_size, conf_d_model)
        
        # --- Conformer Encoder ---
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=conf_d_model,
                n_heads=conf_n_heads,
                d_ff=conf_d_model * 4,
                kernel_size=conf_kernel_size,
                dropout=dropout
            ) for _ in range(conf_n_layers)
        ])
        
        # --- Fully Connected Head ---
        self.fnn = nn.Sequential(
            nn.Linear(conf_d_model, 512),
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
        x = x.permute(0, 2, 1, 3) 
        
        # Apply CNN blocks
        for block in self.cnn_blocks:
            x = block(x)
            
        # x shape: (B, C_out, T, F_out)
        x = x.permute(0, 2, 1, 3) # (B, T, C_out, F_out)
        B, T, C_out, F_out = x.shape
        x = x.reshape(B, T, C_out * F_out)
        
        # Project to Conformer dimension
        x = self.proj(x) # (B, T, d_model)
        
        # Apply Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)
        
        # Apply Head
        logits = self.fnn(x) # (B, T, G*M)
        
        # Reshape to grid
        logits = logits.view(B, T, self.grid_cells, self.num_classes)
        
        return logits
