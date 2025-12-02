import torch
import torch.nn as nn
import torch.nn.functional as F
from model_conformer import ConformerBlock

class Bottleneck(nn.Module):
    """
    Standard ResNet Bottleneck Block.
    Expansion factor = 4.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet50Encoder(nn.Module):
    """
    ResNet50 Encoder modified for Audio Spectrograms.
    
    Modifications:
    1. Input channels: 4 (FOA) instead of 3 (RGB).
    2. Stem Conv1: 3x3 kernel instead of 7x7 to preserve fine details.
    3. Striding: 
       - Time dimension stride is kept at 1 to preserve temporal resolution.
       - Frequency dimension stride is 2 in stem and early layers to reduce dimensionality.
    """
    def __init__(self, in_channels=4, layers=[3, 4, 6, 3]):
        super().__init__()
        self.inplanes = 64
        
        # Stem: 3x3 conv, stride (1, 2) -> preserves time, halves freq
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=(1, 2), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # MaxPool: 3x3, stride (1, 2) -> preserves time, halves freq
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)

        # ResNet Layers
        # Layer 1: 64 -> 256 channels, stride 1
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0], stride=1)
        
        # Layer 2: 256 -> 512 channels, stride (1, 2) -> preserves time, halves freq
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=(1, 2))
        
        # Layer 3: 512 -> 1024 channels, stride (1, 2) -> preserves time, halves freq
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=(1, 2))
        
        # Layer 4: 1024 -> 2048 channels, stride (1, 2) -> preserves time, halves freq
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=(1, 2))
        
        # Total Frequency Stride: 
        # Conv1 (2) * MaxPool (2) * Layer2 (2) * Layer3 (2) * Layer4 (2) = 32
        # Input Freq 64 -> Output Freq 2
        
        self.out_channels = 2048
        self.out_freq = 2 # Assuming input 64 mel bins

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, T, F)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class SELD_ResNet50_Conformer(nn.Module):
    """
    Hybrid Architecture: ResNet50 Encoder + Conformer Temporal Modeling.
    """
    def __init__(self, n_channels=4, n_mels=64, grid_size=(18, 36), num_classes=14,
                 conf_d_model=512, conf_n_heads=8, conf_n_layers=4, conf_kernel_size=31, dropout=0.3):
        super().__init__()
        self.I, self.J = grid_size
        self.grid_cells = self.I * self.J
        self.num_classes = num_classes
        
        # 1. ResNet50 Encoder
        self.encoder = ResNet50Encoder(in_channels=n_channels)
        
        # Calculate encoder output dimension
        # ResNet50 output channels: 2048
        # ResNet50 output freq: n_mels // 32 (e.g., 64 // 32 = 2)
        self.enc_feat_dim = self.encoder.out_channels * (n_mels // 32)
        
        # 2. Projection to Conformer Dimension
        self.proj = nn.Linear(self.enc_feat_dim, conf_d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 3. Conformer Temporal Encoder
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=conf_d_model,
                n_heads=conf_n_heads,
                d_ff=conf_d_model * 4,
                kernel_size=conf_kernel_size,
                dropout=dropout
            ) for _ in range(conf_n_layers)
        ])
        
        # 4. Classification Head
        self.head = nn.Sequential(
            nn.Linear(conf_d_model, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, self.grid_cells * num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C, F)
        Returns:
            logits: (B, T, G, M)
        """
        B, T, C, F = x.shape
        
        # Permute for ResNet: (B, T, C, F) -> (B, C, T, F)
        # Note: We treat Time as Height and Freq as Width for standard Conv2d?
        # No, usually (B, C, H, W). Here H=Time, W=Freq.
        # But our ResNet is designed with stride (1, 2), meaning stride 1 in H (Time) and 2 in W (Freq).
        # So we must pass (B, C, T, F).
        x = x.permute(0, 2, 1, 3) # (B, C, T, F)
        
        # Encoder
        x = self.encoder(x) # (B, 2048, T, F_out)
        
        # Flatten Frequency and Channels
        # x: (B, C_out, T, F_out) -> (B, T, C_out * F_out)
        x = x.permute(0, 2, 1, 3) # (B, T, C_out, F_out)
        B, T, C_out, F_out = x.shape
        x = x.reshape(B, T, C_out * F_out)
        
        # Projection
        x = self.proj(x)
        x = self.dropout(x)
        
        # Conformer
        for block in self.conformer_blocks:
            x = block(x)
            
        # Head
        logits = self.head(x) # (B, T, G*M)
        
        # Reshape to grid
        logits = logits.view(B, T, self.grid_cells, self.num_classes)
        
        return logits
