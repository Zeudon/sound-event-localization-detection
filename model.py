import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    """Standard convolution with BN and SiLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block with residual connection"""
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.cv1 = Conv(in_channels, out_channels, 1, 1, 0)
        self.cv2 = Conv(out_channels, out_channels, 3, 1, 1)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions"""
    def __init__(self, in_channels, out_channels, n_blocks=1, shortcut=True):
        super().__init__()
        hidden_channels = out_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1, 0)
        self.cv2 = Conv(in_channels, hidden_channels, 1, 1, 0)
        self.cv3 = Conv(2 * hidden_channels, out_channels, 1, 1, 0)
        self.m = nn.Sequential(
            *[Bottleneck(hidden_channels, hidden_channels, shortcut) for _ in range(n_blocks)]
        )
    
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1, 0)
        self.cv2 = Conv(hidden_channels * 4, out_channels, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class CSPDarkNet53(nn.Module):
    """CSPDarkNet53 backbone for audio SELD"""
    def __init__(self, in_channels=4, base_channels=64, depth_multiple=1.0, width_multiple=1.0):
        super().__init__()
        
        def get_channels(c):
            return max(round(c * width_multiple), 1)
        
        def get_depth(n):
            return max(round(n * depth_multiple), 1)
        
        self.stem = Conv(in_channels, get_channels(base_channels), 3, 1, 1)
        
        self.stage1 = nn.Sequential(
            Conv(get_channels(64), get_channels(128), 3, 2, 1),
            C3(get_channels(128), get_channels(128), n_blocks=get_depth(3))
        )
        
        self.stage2 = nn.Sequential(
            Conv(get_channels(128), get_channels(256), 3, 2, 1),
            C3(get_channels(256), get_channels(256), n_blocks=get_depth(6))
        )
        
        self.stage3 = nn.Sequential(
            Conv(get_channels(256), get_channels(512), 3, 2, 1),
            C3(get_channels(512), get_channels(512), n_blocks=get_depth(9))
        )
        
        self.stage4 = nn.Sequential(
            Conv(get_channels(512), get_channels(1024), 3, 2, 1),
            C3(get_channels(1024), get_channels(1024), n_blocks=get_depth(3)),
            SPPF(get_channels(1024), get_channels(1024))
        )
        
        self.out_channels = [
            get_channels(128),
            get_channels(256),
            get_channels(512),
            get_channels(1024)
        ]
    
    def forward(self, x):
        x = self.stem(x)
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return [p2, p3, p4, p5]


class SMRSELDWithCSPDarkNet(nn.Module):
    """
    SMR-SELD model with a CSPDarkNet53 backbone and proper (I, J) grid pooling.

    Parameters
    ----------
    n_channels : int
        Number of input channels (4 for FOA).
    grid_size : tuple of int
        (I, J) specifying number of elevation and azimuth bins.
    num_classes : int
        Number of event classes including background.
    use_small : bool
        If True, use a reduced backbone (depth and width multipliers).
    """
    def __init__(self, n_channels=4, grid_size=(18, 36), num_classes=14, use_small=True):
        super().__init__()
        self.I, self.J = grid_size
        self.grid_cells = self.I * self.J
        self.num_classes = num_classes

        # Backbone
        if use_small:
            self.backbone = CSPDarkNet53(
                in_channels=n_channels,
                depth_multiple=0.33,
                width_multiple=0.5
            )
        else:
            self.backbone = CSPDarkNet53(in_channels=n_channels)

        # Multi-scale fusion: use P3, P4, P5 as in many detection models
        c3, c4, c5 = self.backbone.out_channels[1:]
        self.reduce_p3 = nn.Conv2d(c3, 256, kernel_size=1)
        self.reduce_p4 = nn.Conv2d(c4, 256, kernel_size=1)
        self.reduce_p5 = nn.Conv2d(c5, 256, kernel_size=1)

        # Fuse 3 scales → 256 channels
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(256 * 3, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )

        # Grid pooling: map fused feature map → (I, J) grid
        # This directly produces one feature vector per spatial cell.
        self.grid_pool = nn.AdaptiveAvgPool2d((self.I, self.J))

        # Per-cell classification head (shared across grid cells)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),       # feature normalization
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x: (B, T, C, F)  – batch, frames, channels, mel bins
        Returns:
            probs: (B, T, G, M) – per-grid-cell class probabilities
        """
        B, T, C, F_freq = x.shape

        # Treat each time frame independently in the CNN:
        # (B*T, C, F, 1) so F is 'height' and width=1
        x = x.reshape(B * T, C, F_freq, 1)

        # Backbone multi-scale features
        features = self.backbone(x)  # [p2, p3, p4, p5]
        p3, p4, p5 = features[1], features[2], features[3]

        # Channel reduction to 256 per scale
        p3 = self.reduce_p3(p3)
        p4 = self.reduce_p4(p4)
        p5 = self.reduce_p5(p5)

        # Upsample P4, P5 to P3's spatial size
        target_size = p3.shape[2:]  # (H, W)
        p4 = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)
        p5 = F.interpolate(p5, size=target_size, mode='bilinear', align_corners=False)

        # Multi-scale fusion: concat along channels then fuse to 256-channel feature map
        fused = torch.cat([p3, p4, p5], dim=1)  # (B*T, 256*3, H, W)
        fused = self.conv_fuse(fused)           # (B*T, 256, H, W)

        # Pool fused features to the DOA grid: (I, J)
        # Result: (B*T, 256, I, J)
        grid_feat = self.grid_pool(fused)

        # Reshape to per-cell feature vectors: (B*T, I*J, 256)
        grid_feat = grid_feat.view(B * T, 256, self.grid_cells).transpose(1, 2)

        # Optional L2 normalisation per cell (stabilises CL & AIUR)
        grid_feat = F.normalize(grid_feat, p=2, dim=-1)

        # Apply shared classifier per cell: flatten cells into batch dimension
        logits = self.classifier(grid_feat)  # (B*T, G, M)

        # Reshape back to (B, T, G, M)
        logits = logits.view(B, T, self.grid_cells, self.num_classes)

        # Return logits directly for CrossEntropyLoss
        # probs = F.softmax(logits, dim=-1)

        return logits
