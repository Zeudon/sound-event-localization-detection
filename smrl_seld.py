#!/usr/bin/env python3
"""
SMR-SELD Training Script for HPC
================================

This script is a direct conversion of SMR_SELD.ipynb with ONLY the following changes:
1. CUDA optimizations for faster training
2. Logging instead of print statements

ALL model logic, preprocessing, training loops, and loss functions are IDENTICAL to the notebook.
Configuration is managed through the Config class.
"""

import os
import sys
import math
import random
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchaudio
from glob import glob
import gc


# ==============================================================================
# LOGGING SETUP (replacing print statements)
# ==============================================================================
def setup_logging(log_dir='logs', experiment_name='smr_seld'):
    """Setup comprehensive logging for HPC environment"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{experiment_name}_{timestamp}.log')
    
    # Create logger
    logger = logging.getLogger('SMR_SELD')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicates when re-running cells
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file

logger, log_file = setup_logging()

# ==============================================================================
# DEVICE SETUP (CUDA optimization #1)
# ==============================================================================
def get_device():
    """Get available device with CUDA support"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # CUDA optimization: Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        logger.info("CuDNN benchmarking enabled for optimized performance")
    else:
        device = torch.device('cpu')
        logger.warning("CUDA not available. Using CPU. Training will be slower.")
    
    return device


def safe_torch_load(path, map_location=None):
    """
    Safely load PyTorch checkpoint with compatibility for different PyTorch versions.
    PyTorch 2.6+ requires weights_only parameter for non-weight checkpoints.
    """
    try:
        # Try loading with weights_only=False for PyTorch 2.6+
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Fall back to old API for PyTorch < 2.6
        return torch.load(path, map_location=map_location)


DEVICE = get_device()

# ==============================================================================
# CONFIGURATION CLASS (IDENTICAL TO NOTEBOOK)
# ==============================================================================
class Config:
    """Configuration class for SMR-SELD training"""
    
    # Paths (relative to script location)
    BASE_PATH = Path(__file__).parent
    AUDIO_PATH = BASE_PATH / "foa_dev"
    METADATA_PATH = BASE_PATH / "metadata_dev"
    OUTPUT_PATH = BASE_PATH / "outputs"
    CHECKPOINT_PATH = BASE_PATH / "checkpoints"
    
    # Dataset - Use full dataset or single file for testing
    USE_FULL_DATASET = True
    TRAIN_AUDIO_FILE = "fold3_room21_mix001.wav"
    TRAIN_META_FILE = "fold3_room21_mix001.csv"
    TEST_AUDIO_FILE = "fold4_room23_mix001.wav"
    TEST_META_FILE = "fold4_room23_mix001.csv"
    
    # STARSS22 Classes
    STARSS22_CLASSES = {
        0: 'Female speech, woman speaking',
        1: 'Male speech, man speaking',
        2: 'Clapping',
        3: 'Telephone',
        4: 'Laughter',
        5: 'Domestic sounds',
        6: 'Walk, footsteps',
        7: 'Door, open or close',
        8: 'Music',
        9: 'Musical instrument',
        10: 'Water tap, faucet',
        11: 'Bell',
        12: 'Knock',
        13: 'Background'
    }
    
    # Model
    NUM_CLASSES = 14
    N_CHANNELS = 4
    
    # Training hyperparameters
    NUM_EPOCHS = 30
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    LR_DECAY_FACTOR = 0.5
    LR_DECAY_PATIENCE = 5
    WEIGHT_DECAY = 1e-4
    
    # Loss weights
    W_CLASS = 1.0
    W_AIUR = 1.0
    W_CL = 1.0
    
    # Early stopping
    PATIENCE = 20
    MIN_DELTA = 1e-4
    
    # Checkpointing
    SAVE_EVERY_N_EPOCHS = 5
    KEEP_LAST_N_CHECKPOINTS = 3
    
    # Signal Processing
    SPECTROGRAM_N_FFT = int(0.04*24000)
    SPECTROGRAM_HOP_LENGTH = int(0.02*24000) 
    N_MELS = 64
    SR = 24000
    
    # Dataset windowing
    WINDOW_LENGTH = int(5*24000)
    HOP_LENGTH = int(1*24000)
    
    # 3D to 2D Mapping
    I = None
    J = None
    GRID_CELL_DEGREES = 10
    
    def __init__(self):
        # Create directories
        self.OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
        self.CHECKPOINT_PATH.mkdir(exist_ok=True, parents=True)
        
        # Build full paths
        self.TRAIN_AUDIO_PATH = self.AUDIO_PATH / "dev-train-sony" / self.TRAIN_AUDIO_FILE
        self.TRAIN_META_PATH = self.METADATA_PATH / "dev-train-sony" / self.TRAIN_META_FILE
        self.TEST_AUDIO_PATH = self.AUDIO_PATH / "dev-test-sony" / self.TEST_AUDIO_FILE
        self.TEST_META_PATH = self.METADATA_PATH / "dev-test-sony" / self.TEST_META_FILE
        
        # Dataset directories
        self.SONY_TRAIN_DIR = self.AUDIO_PATH / "dev-train-sony"
        self.SONY_TEST_DIR = self.AUDIO_PATH / "dev-test-sony"
        self.SONY_TRAIN_META_DIR = self.METADATA_PATH / "dev-train-sony"
        self.SONY_TEST_META_DIR = self.METADATA_PATH / "dev-test-sony"
        self.TAU_TRAIN_DIR = self.AUDIO_PATH / "dev-train-tau"
        self.TAU_TEST_DIR = self.AUDIO_PATH / "dev-test-tau"
        self.TAU_TRAIN_META_DIR = self.METADATA_PATH / "dev-train-tau"
        self.TAU_TEST_META_DIR = self.METADATA_PATH / "dev-test-tau"


# ==============================================================================
# DATA LOADING (IDENTICAL TO NOTEBOOK)
# ==============================================================================
def load_files():
    """Load audio and metadata files based on configuration"""
    if config.USE_FULL_DATASET:
        # Load all audio files
        sony_train_audio = sorted(glob(str(config.SONY_TRAIN_DIR / "*.wav")))
        tau_train_audio = sorted(glob(str(config.TAU_TRAIN_DIR / "*.wav")))
        sony_test_audio = sorted(glob(str(config.SONY_TEST_DIR / "*.wav")))
        tau_test_audio = sorted(glob(str(config.TAU_TEST_DIR / "*.wav")))
        
        # Match metadata files to audio files by basename
        def get_matching_metadata(audio_files, meta_dir):
            """Get metadata files matching audio files by basename"""
            meta_files = []
            for audio_file in audio_files:
                # Get basename without extension (e.g., fold3_room21_mix001)
                basename = Path(audio_file).stem
                # Build corresponding metadata path
                meta_file = meta_dir / f"{basename}.csv"
                if meta_file.exists():
                    meta_files.append(str(meta_file))
                else:
                    raise FileNotFoundError(f"Metadata file not found: {meta_file}")
            return meta_files
        
        # Get matching metadata files
        sony_train_meta = get_matching_metadata(sony_train_audio, config.SONY_TRAIN_META_DIR)
        tau_train_meta = get_matching_metadata(tau_train_audio, config.TAU_TRAIN_META_DIR)
        sony_test_meta = get_matching_metadata(sony_test_audio, config.SONY_TEST_META_DIR)
        tau_test_meta = get_matching_metadata(tau_test_audio, config.TAU_TEST_META_DIR)
        
        # Combine training and testing files
        train_audio_files = sony_train_audio + tau_train_audio
        train_meta_files = sony_train_meta + tau_train_meta
        test_audio_files = sony_test_audio + tau_test_audio
        test_meta_files = sony_test_meta + tau_test_meta
    else:
        # Load single training file
        train_audio_files = [str(config.TRAIN_AUDIO_PATH)]
        train_meta_files = [str(config.TRAIN_META_PATH)]
        
        # Load single testing file
        test_audio_files = [str(config.TEST_AUDIO_PATH)]
        test_meta_files = [str(config.TEST_META_PATH)]
    
    return train_audio_files, train_meta_files, test_audio_files, test_meta_files


# ==============================================================================
# COORDINATE TRANSFORMATION (IDENTICAL TO NOTEBOOK)
# ==============================================================================
def polar_to_grid(phi, theta, I=None, J=None, cell_size_deg=None):
    """Convert polar coordinates to grid indices"""
    if (I is None or J is None) and cell_size_deg is not None:
        I = int(180 // cell_size_deg)
        J = int(360 // cell_size_deg)
    elif I is None or J is None:
        raise ValueError("Either provide (I, J) or cell_size_deg for polar_to_grid")

    # Normalize azimuth and elevation to [0,1]
    phi_norm = (phi + 180.0) / 360.0
    theta_norm = (theta + 90.0) / 180.0
    j = int(np.clip(phi_norm * J, 0, J - 1))
    i = int(np.clip(theta_norm * I, 0, I - 1))
    return i, j


# ==============================================================================
# AUDIO LOADING (IDENTICAL TO NOTEBOOK)
# ==============================================================================
def load_audio(audio_path):
    """Load multi-channel audio file using torchaudio"""
    waveform, sample_rate = torchaudio.load(audio_path)
    
    if waveform.shape[0] != 4:
        logger.warning(f"Expected 4 channels but got {waveform.shape[0]} channels in {audio_path}")
    
    return waveform, sample_rate


def audio_to_mel_spectrogram(waveform, sample_rate, n_fft=None, hop_length=None, n_mels=None):
    """Convert multi-channel audio waveform to mel spectrogram"""
    # Use config defaults if not provided
    if n_fft is None:
        n_fft = config.SPECTROGRAM_N_FFT
    if hop_length is None:
        hop_length = config.SPECTROGRAM_HOP_LENGTH
    if n_mels is None:
        n_mels = config.N_MELS
    
    # Create mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Process each channel separately
    mel_specs = []
    for channel_idx in range(waveform.shape[0]):
        channel_waveform = waveform[channel_idx:channel_idx+1, :]  # Keep dimension (1, samples)
        mel_spec = mel_transform(channel_waveform)  # Shape: (1, n_mels, time_frames)
        mel_specs.append(mel_spec)
    
    # Stack all channels: (channels, n_mels, time_frames)
    mel_spec_multichannel = torch.cat(mel_specs, dim=0)
    
    # Convert to log scale (dB)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec_multichannel)
    
    return mel_spec_db


def metadata_to_labels(metadata_path, audio_duration, sample_rate=24000, I=None, J=None, 
                        cell_size_deg=None, num_classes=14):
    """Convert metadata file to target labels for 2D grid representation"""

    if cell_size_deg is None:
        cell_size_deg = config.GRID_CELL_DEGREES
    
    # Step 1: Calculate total number of frames (20ms per frame)
    frame_duration_ms = 20  # 20ms per frame in final representation
    metadata_frame_duration_ms = 100  # 100ms per frame in metadata
    frames_per_metadata_frame = metadata_frame_duration_ms // frame_duration_ms  # = 5
    
    # Total number of frames for the audio
    total_frames = int((audio_duration * 1000) / frame_duration_ms)
    
    if (I is None or J is None) and cell_size_deg is not None:
        I = int(180 // cell_size_deg)
        J = int(360 // cell_size_deg)
    elif I is None or J is None:
        raise ValueError("Either provide (I, J) or cell_size_deg for grid dimensions")
    
    total_cells = I * J
    
    # Initialize labels tensor: [T', I*J, M] with all zeros
    labels = torch.zeros((total_frames, total_cells, num_classes), dtype=torch.float32)
    
    df = pd.read_csv(metadata_path, header=None)
    
    # Track which cells have active events for each frame
    active_cells_per_frame = [set() for _ in range(total_frames)]
    
    # Process each row in metadata
    for _, row in df.iterrows():
        metadata_frame = int(row.iloc[0])  # Frame number from metadata
        active_class = int(row.iloc[1])     # Active class index
        source_num = int(row.iloc[2])       # Source number (not used in labeling)
        azimuth = int(row.iloc[3])          # Azimuth in degrees
        elevation = int(row.iloc[4])        # Elevation in degrees
        
        # Metadata frame t corresponds to frames t*5 to t*5+4 in final representation
        start_frame = metadata_frame * frames_per_metadata_frame
        end_frame = start_frame + frames_per_metadata_frame
        
        end_frame = min(end_frame, total_frames)
        
        i, j = polar_to_grid(azimuth, elevation, I=I, J=J)
        cell_idx = i * J + j  # Flatten 2D grid to 1D index
        
        # Step 4: Set active class for this cell across the time frames
        for t in range(start_frame, end_frame):
            labels[t, cell_idx, active_class] = 1.0
            active_cells_per_frame[t].add(cell_idx)
    
    # Step 5: Set background class (index 13) for cells with no active events
    for t in range(total_frames):
        for cell_idx in range(total_cells):
            if cell_idx not in active_cells_per_frame[t]:
                labels[t, cell_idx, num_classes - 1] = 1.0
    
    return labels, I, J


# ==============================================================================
# DATASET CLASS (IDENTICAL TO NOTEBOOK)
# ==============================================================================
class SELDDataset(Dataset):
    def __init__(self, audio_files, metadata_files, num_classes=14):
        """
        Initialize SELD Dataset with windowing.
        
        This dataset:
        1. Loads all audio files and computes spectrograms + labels
        2. Concatenates all spectrograms and labels into single tensors
        3. Segments concatenated data into windows (5s window, 1s hop)
        4. Pads final window if needed
        """
        assert len(audio_files) == len(metadata_files), \
            "Number of audio files must match number of metadata files"
        
        self.audio_files = audio_files
        self.metadata_files = metadata_files
        self.sample_rate = config.SR
        self.n_fft = config.SPECTROGRAM_N_FFT
        self.spectrogram_hop_length = config.SPECTROGRAM_HOP_LENGTH
        self.n_mels = config.N_MELS
        self.cell_size_deg = config.GRID_CELL_DEGREES
        self.num_classes = num_classes
        
        self.I = int(180 // self.cell_size_deg)
        self.J = int(360 // self.cell_size_deg)
        self.total_cells = self.I * self.J
        
        self.window_length_samples = config.WINDOW_LENGTH  # 5s in samples
        self.hop_length_samples = config.HOP_LENGTH  # 1s in samples
        
        # Convert to spectrogram frames
        # Each spectrogram frame represents spectrogram_hop_length samples
        self.window_length_frames = int(self.window_length_samples / self.spectrogram_hop_length)
        self.hop_length_frames = int(self.hop_length_samples / self.spectrogram_hop_length)
        
        logger.info(f"SELDDataset initialization started...")
        logger.info(f"  Files: {len(audio_files)} audio files")
        logger.info(f"  Grid: {self.I}x{self.J} = {self.total_cells} cells")
        logger.info(f"  Window: {self.window_length_frames} frames ({self.window_length_samples / self.sample_rate:.1f}s)")
        logger.info(f"  Hop: {self.hop_length_frames} frames ({self.hop_length_samples / self.sample_rate:.1f}s)")
        
        self._load_and_concatenate_all()
        self._create_windows()
        logger.info(f"SELDDataset initialized with {len(self.windows)} windows")
    
    def _load_and_concatenate_all(self):
        """Load all files, compute spectrograms and labels, then concatenate."""
        all_spectrograms = []
        all_labels = []
        
        logger.info("Loading and processing all audio files...")
        for idx, (audio_path, metadata_path) in enumerate(tqdm(
            zip(self.audio_files, self.metadata_files),
            total=len(self.audio_files),
            desc="Processing files"
        )):
            try:
                waveform, sr = load_audio(audio_path)
                mel_spec = audio_to_mel_spectrogram(
                    waveform, 
                    sr,
                    n_fft=self.n_fft,
                    hop_length=self.spectrogram_hop_length,
                    n_mels=self.n_mels
                )  # Shape: (4, n_mels, time_frames)
                audio_duration = waveform.shape[1] / sr
                labels, _, _ = metadata_to_labels(
                    metadata_path,
                    audio_duration,
                    sample_rate=sr,
                    I=self.I,
                    J=self.J,
                    cell_size_deg=self.cell_size_deg,
                    num_classes=self.num_classes
                )  # Shape: (time_frames, I*J, num_classes)
                
                mel_time_frames = mel_spec.shape[2]
                label_time_frames = labels.shape[0]
                
                if mel_time_frames != label_time_frames:
                    min_frames = min(mel_time_frames, label_time_frames)
                    mel_spec = mel_spec[:, :, :min_frames]
                    labels = labels[:min_frames, :, :]
                
                all_spectrograms.append(mel_spec)
                all_labels.append(labels)
                
            except Exception as e:
                logger.error(f"Error processing file {idx} ({audio_path}): {str(e)}")
                raise
        
        # Concatenate along time dimension
        self.concatenated_spectrograms = torch.cat(all_spectrograms, dim=2)  # (4, n_mels, T)
        self.concatenated_labels = torch.cat(all_labels, dim=0)  # (T, I*J, num_classes)
        
        self.total_frames = self.concatenated_spectrograms.shape[2]
        logger.info(f"Concatenated data: {self.total_frames} total frames")
        logger.info(f"  Spectrograms shape: {self.concatenated_spectrograms.shape}")
        logger.info(f"  Labels shape: {self.concatenated_labels.shape}")
    
    def _create_windows(self):
        """Segment concatenated data into windows with overlap."""
        self.windows = []
        
        start_frame = 0
        window_idx = 0
        
        while start_frame < self.total_frames:
            end_frame = start_frame + self.window_length_frames
            
            # Extract window
            if end_frame <= self.total_frames:
                # Normal window - no padding needed
                window_spec = self.concatenated_spectrograms[:, :, start_frame:end_frame]
                window_labels = self.concatenated_labels[start_frame:end_frame, :, :]
            else:
                # Last window - needs padding
                actual_frames = self.total_frames - start_frame

                window_spec = self.concatenated_spectrograms[:, :, start_frame:]
                window_labels = self.concatenated_labels[start_frame:, :, :]
                
                # Pad to window_length_frames
                pad_frames = self.window_length_frames - actual_frames
                
                # Pad spectrograms: (4, n_mels, time) -> pad time dimension
                spec_pad = torch.zeros((4, self.n_mels, pad_frames), dtype=window_spec.dtype)
                window_spec = torch.cat([window_spec, spec_pad], dim=2)
                
                # Pad labels: (time, I*J, num_classes) -> pad time dimension
                # Set background class (index 13) for padded frames
                label_pad = torch.zeros((pad_frames, self.total_cells, self.num_classes), dtype=window_labels.dtype)
                label_pad[:, :, self.num_classes - 1] = 1.0  # Set background class
                window_labels = torch.cat([window_labels, label_pad], dim=0)
            
            # Transpose spectrogram from [C, F, T] to [T, C, F]
            window_spec = window_spec.permute(2, 0, 1)  # (T, C, F)
            
            # Store window
            self.windows.append({
                'spectrogram': window_spec,
                'labels': window_labels,
                'window_idx': window_idx,
                'start_frame': start_frame,
                'end_frame': min(end_frame, self.total_frames)
            })

            start_frame += self.hop_length_frames
            window_idx += 1
        
        logger.info(f"Created {len(self.windows)} windows")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Get a single window from the dataset.
        Returns:
            spectrogram: Mel spectrogram tensor of shape (window_length_frames, 4, n_mels) - [T, C, F]
            labels: Target labels tensor of shape (window_length_frames, I*J, num_classes) - [T, I*J, M]
        """
        window = self.windows[idx]
        return window['spectrogram'], window['labels']


# ==============================================================================
# MODEL ARCHITECTURE (IDENTICAL TO NOTEBOOK)
# ==============================================================================
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



# ==============================================================================
# LOSS FUNCTION (IDENTICAL TO NOTEBOOK)
# ==============================================================================
class SMRSELDLoss(nn.Module):
    """Complete SMR-SELD loss function with three components"""

    def __init__(self, w_class=1.0, w_aiur=0.5, w_cl=0.5, grid_size=None, class_weights=None):
        super().__init__()
        self.w_class = w_class
        self.w_aiur = w_aiur
        self.w_cl = w_cl
        self.eps = 1e-10  # Small epsilon for numerical stability
        if grid_size is not None:
            self.I, self.J = grid_size
        else:
            self.I = self.J = None
            
        # Initialize CrossEntropyLoss with weights if provided
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
    
    def class_ce_loss(self, y_pred, y_true):
        """Class-wise Cross Entropy loss
        y_pred: (B, T, G, M) - Logits
        y_true: (B, T, G, M) - One-hot encoded targets
        """
        # Convert one-hot targets to class indices
        y_true_indices = torch.argmax(y_true, dim=-1) # (B, T, G)
        
        # Reshape for CrossEntropyLoss: (N, C) vs (N)
        # Flatten batch, time, and grid dimensions
        B, T, G, M = y_pred.shape
        y_pred_flat = y_pred.view(-1, M) # (B*T*G, M)
        y_true_flat = y_true_indices.view(-1) # (B*T*G)
        
        ce_loss = self.ce_loss(y_pred_flat, y_true_flat)
        return ce_loss
    
    def aiur_loss(self, y_pred, y_true):
        """Area Intersection Union Ratio (AIUR) loss computed per frame and batch.
        """
        B, T, G, M = y_pred.shape
        
        # Background class is the last index (index 13 for 14 classes)
        background_idx = M - 1

        y_pred_class = torch.argmax(y_pred, dim=-1)  # (B, T, G)
        y_true_class = torch.argmax(y_true, dim=-1)  # (B, T, G)
        
        # Create binary masks: 1 for event cells, 0 for background
        pred_event_mask = (y_pred_class != background_idx).float()  # (B, T, G)
        true_event_mask = (y_true_class != background_idx).float()  # (B, T, G)

        intersection = (pred_event_mask * true_event_mask).sum(dim=-1)  # (B, T) - sum over grid cells

        pred_count = pred_event_mask.sum(dim=-1)  # (B, T)
        true_count = true_event_mask.sum(dim=-1)  # (B, T)
        union = pred_count + true_count - intersection  # (B, T)

        epsilon = 1e-8
        iou = intersection / (union + epsilon)  # (B, T)
        
        # Handle edge case where both pred and true have no events (union = 0)
        # In this case, IoU should be 1.0 (perfect match of empty sets)
        iou = torch.where(union > 0, iou, torch.ones_like(iou))

        avg_iou = iou.mean()
        aiur_loss_value = 1.0 - avg_iou
        
        return aiur_loss_value
    
    def converging_localization_loss(self, y_pred, y_true):
        """
        Converging localization loss.

        y_pred, y_true: (B, T, G, M)
        Uses only non-background probability and operates on the (I, J) grid.
        """
        B, T, G, M = y_pred.shape

        # Grid dimensions (I, J) – infer if not provided
        if self.I is not None and self.J is not None:
            I, J = self.I, self.J
        else:
            I = J = int(math.sqrt(G))

        # Reshape to (B, T, I, J)
        y_pred_grid = y_pred.view(B, T, I, J, M)
        y_true_grid = y_true.view(B, T, I, J, M)

        # True/Pred non-background “activity” per cell
        true_nonbg = y_true_grid[..., :-1].sum(dim=-1)  # (B, T, I, J)
        pred_nonbg = y_pred_grid[..., :-1].sum(dim=-1)  # (B, T, I, J)

        # Count background and non-background cells per frame
        N_bac = (true_nonbg < 0.01).sum(dim=(2, 3), keepdim=True).float()   # (B, T, 1, 1)
        N_non = (true_nonbg > 0.01).sum(dim=(2, 3), keepdim=True).float()   # (B, T, 1, 1)

        # Step 1: transform targets: y'_ij = 1 (background), -N_bac/N_non (events)
        y_prime = torch.ones_like(true_nonbg)            # (B, T, I, J)
        ratio = -(N_bac / (N_non + self.eps))            # (B, T, 1, 1)
        y_prime = torch.where(true_nonbg > 0.01,
                              ratio.expand_as(true_nonbg),
                              y_prime)

        # Step 2: neighbourhood density (Eq. 5) with circular padding
        y_prime_padded = F.pad(y_prime, (1, 1, 1, 1), mode='circular')  # (B, T, I+2, J+2)

        diff_sum = torch.zeros_like(y_prime)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                neighbor = y_prime_padded[:, :, 1+di:I+1+di, 1+dj:J+1+dj]
                diff_sum += (neighbor - y_prime)

        avg_diff = diff_sum / 8.0
        y_at = y_prime + avg_diff   # (B, T, I, J)

        # Step 3: apply only on frames that have events
        has_events_mask = (N_non > 0).float()           # (B, T, 1, 1)

        # Multiply predicted non-background by attention map
        # and normalise by (num event frames * grid cells)
        weighted = (pred_nonbg * y_at) * has_events_mask  # (B, T, I, J)
        denom = (has_events_mask.sum() * I * J) + self.eps

        loss = weighted.sum() / denom
        return loss


    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # y_pred are logits now
        loss_class = self.class_ce_loss(y_pred, y_true)
        
        # For other losses (AIUR, CL), we need probabilities
        # y_pred_probs = F.softmax(y_pred, dim=-1)
        
        # loss_aiur = self.aiur_loss(y_pred_probs, y_true)
        # loss_cl = self.converging_localization_loss(y_pred_probs, y_true)
        
        # total_loss = (self.w_class * loss_class +
        #               self.w_aiur * loss_aiur +
        #               self.w_cl * loss_cl)
        total_loss = self.w_class * loss_class  
        breakdown = {
            'class_ce': float(loss_class.item()),
            # 'aiur': float(loss_aiur.item()),
            # 'cl': float(loss_cl.item())
        }
        return total_loss, breakdown


# ==============================================================================
# LOSS VISUALIZATION FUNCTION
# ==============================================================================
def visualize_loss_components(
    y_pred, 
    y_true, 
    criterion, 
    epoch, 
    save_dir='loss_visualizations',
    frame_idx=None
):
    """
    Visualize what each loss component does for a single frame with events.
    
    Args:
        y_pred: Model predictions (B, T, G, M)
        y_true: Ground truth labels (B, T, G, M)
        criterion: SMRSELDLoss instance
        epoch: Current epoch number
        save_dir: Directory to save visualizations
        frame_idx: Optional specific frame index to visualize
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Move to CPU for visualization
    y_pred = y_pred.detach().cpu()
    
    # Apply softmax to logits for visualization
    y_pred = F.softmax(y_pred, dim=-1)
    
    y_true = y_true.detach().cpu()
    
    B, T, G, M = y_pred.shape
    I, J = criterion.I, criterion.J
    
    # Find a frame with sufficient events (> 5 active cells)
    y_true_class = torch.argmax(y_true, dim=-1)  # (B, T, G)
    background_idx = M - 1
    
    if frame_idx is None:
        # Find frame with most events
        true_event_mask = (y_true_class != background_idx).float()  # (B, T, G)
        event_counts = true_event_mask.sum(dim=-1)  # (B, T)
        
        # Find frame with at least 1 events
        valid_frames = (event_counts >= 1).nonzero(as_tuple=False)
        
        if len(valid_frames) == 0:
            logger.warning("No frames with sufficient events found for visualization")
            return
        
        # Pick the frame with most events
        b_idx, t_idx = valid_frames[event_counts[valid_frames[:, 0], valid_frames[:, 1]].argmax()]
        b_idx, t_idx = int(b_idx), int(t_idx)
    else:
        b_idx, t_idx = 0, frame_idx
    
    # Extract single frame
    pred_frame = y_pred[b_idx, t_idx]  # (G, M)
    true_frame = y_true[b_idx, t_idx]  # (G, M)
    
    # Reshape to grid
    pred_grid = pred_frame.view(I, J, M)  # (I, J, M)
    true_grid = true_frame.view(I, J, M)  # (I, J, M)
    
    # Get class predictions
    pred_class = torch.argmax(pred_grid, dim=-1)  # (I, J)
    true_class = torch.argmax(true_grid, dim=-1)  # (I, J)
    
    # Get non-background activity
    pred_nonbg = pred_grid[..., :-1].sum(dim=-1)  # (I, J)
    true_nonbg = true_grid[..., :-1].sum(dim=-1)  # (I, J)
    
    # ========== Compute AIUR Loss Components ==========
    pred_event_mask = (pred_class != background_idx).float()  # (I, J)
    true_event_mask = (true_class != background_idx).float()  # (I, J)
    
    intersection_map = pred_event_mask * true_event_mask  # (I, J)
    union_map = (pred_event_mask + true_event_mask).clamp(0, 1)  # (I, J)
    
    # ========== Compute Converging Localization Loss Components ==========
    # Reshape back to (1, 1, I, J, M) for CL loss computation
    pred_grid_batch = pred_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, I, J, M)
    true_grid_batch = true_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, I, J, M)
    
    # Get non-background for CL computation
    true_nonbg_cl = true_grid_batch[..., :-1].sum(dim=-1).squeeze(0).squeeze(0)  # (I, J)
    pred_nonbg_cl = pred_grid_batch[..., :-1].sum(dim=-1).squeeze(0).squeeze(0)  # (I, J)
    
    # Count background and non-background cells
    N_bac = (true_nonbg_cl < 0.01).sum().float()
    N_non = (true_nonbg_cl > 0.01).sum().float()
    
    # Step 1: transform targets
    y_prime = torch.ones_like(true_nonbg_cl)
    if N_non > 0:
        ratio = -(N_bac / N_non)
        y_prime = torch.where(true_nonbg_cl > 0.01, ratio, y_prime)
    
    # Step 2: neighbourhood density with circular padding
    y_prime_padded = F.pad(y_prime.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='circular')
    y_prime_padded = y_prime_padded.squeeze(0).squeeze(0)  # (I+2, J+2)
    
    diff_sum = torch.zeros_like(y_prime)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbor = y_prime_padded[1+di:I+1+di, 1+dj:J+1+dj]
            diff_sum += (neighbor - y_prime)
    
    avg_diff = diff_sum / 8.0
    y_at = y_prime + avg_diff  # (I, J) - attention map
    
    # CL loss contribution per cell
    cl_contribution = pred_nonbg_cl * y_at  # (I, J)
    
    # ========== Create Visualization ==========
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Color maps
    cmap_activity = 'YlOrRd'
    cmap_class = 'tab20'
    cmap_diverging = 'RdBu_r'
    
    # Row 1: Ground Truth
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(true_nonbg.numpy(), cmap=cmap_activity, aspect='auto')
    ax1.set_title(f'Ground Truth Activity\n(Non-background Sum)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Azimuth bins')
    ax1.set_ylabel('Elevation bins')
    plt.colorbar(im1, ax=ax1, label='Activity')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(true_class.numpy(), cmap=cmap_class, aspect='auto', vmin=0, vmax=M-1)
    ax2.set_title(f'Ground Truth Classes\n({int(true_event_mask.sum())} active cells)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Azimuth bins')
    ax2.set_ylabel('Elevation bins')
    plt.colorbar(im2, ax=ax2, label='Class ID', ticks=range(M))
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(true_event_mask.numpy(), cmap='Greys', aspect='auto', vmin=0, vmax=1)
    ax3.set_title('Ground Truth Event Mask\n(1=Event, 0=Background)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Azimuth bins')
    ax3.set_ylabel('Elevation bins')
    plt.colorbar(im3, ax=ax3, label='Mask')
    
    # Statistics for GT
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    gt_stats = f"""Ground Truth Statistics:
    
Total Cells: {I * J}
Active Cells: {int(true_event_mask.sum())}
Background Cells: {int((1 - true_event_mask).sum())}
Activity Range: [{true_nonbg.min():.3f}, {true_nonbg.max():.3f}]
N_bac: {N_bac:.0f}
N_non: {N_non:.0f}
    """
    ax4.text(0.1, 0.5, gt_stats, fontsize=11, verticalalignment='center', family='monospace')
    
    # Row 2: Predictions
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(pred_nonbg.numpy(), cmap=cmap_activity, aspect='auto')
    ax5.set_title(f'Predicted Activity\n(Non-background Sum)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Azimuth bins')
    ax5.set_ylabel('Elevation bins')
    plt.colorbar(im5, ax=ax5, label='Activity')
    
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(pred_class.numpy(), cmap=cmap_class, aspect='auto', vmin=0, vmax=M-1)
    ax6.set_title(f'Predicted Classes\n({int(pred_event_mask.sum())} active cells)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Azimuth bins')
    ax6.set_ylabel('Elevation bins')
    plt.colorbar(im6, ax=ax6, label='Class ID', ticks=range(M))
    
    ax7 = fig.add_subplot(gs[1, 2])
    im7 = ax7.imshow(pred_event_mask.numpy(), cmap='Greys', aspect='auto', vmin=0, vmax=1)
    ax7.set_title('Predicted Event Mask\n(1=Event, 0=Background)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Azimuth bins')
    ax7.set_ylabel('Elevation bins')
    plt.colorbar(im7, ax=ax7, label='Mask')
    
    # Statistics for Predictions
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    pred_stats = f"""Prediction Statistics:
    
Total Cells: {I * J}
Active Cells: {int(pred_event_mask.sum())}
Background Cells: {int((1 - pred_event_mask).sum())}
Activity Range: [{pred_nonbg.min():.3f}, {pred_nonbg.max():.3f}]
Confidence: {pred_grid.max(dim=-1)[0].mean():.3f}
    """
    ax8.text(0.1, 0.5, pred_stats, fontsize=11, verticalalignment='center', family='monospace')
    
    # Row 3: Loss Components
    ax9 = fig.add_subplot(gs[2, 0])
    im9 = ax9.imshow(intersection_map.numpy(), cmap='Greens', aspect='auto', vmin=0, vmax=1)
    ax9.set_title('AIUR: Intersection Map\n(Correct Predictions)', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Azimuth bins')
    ax9.set_ylabel('Elevation bins')
    plt.colorbar(im9, ax=ax9, label='Intersection')
    
    ax10 = fig.add_subplot(gs[2, 1])
    im10 = ax10.imshow(union_map.numpy(), cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax10.set_title('AIUR: Union Map\n(All Predicted or True Events)', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Azimuth bins')
    ax10.set_ylabel('Elevation bins')
    plt.colorbar(im10, ax=ax10, label='Union')
    
    ax11 = fig.add_subplot(gs[2, 2])
    im11 = ax11.imshow(y_at.numpy(), cmap=cmap_diverging, aspect='auto')
    ax11.set_title('CL: Attention Map (y_at)\n(Neighborhood-aware Targets)', fontsize=12, fontweight='bold')
    ax11.set_xlabel('Azimuth bins')
    ax11.set_ylabel('Elevation bins')
    plt.colorbar(im11, ax=ax11, label='Attention')
    
    ax12 = fig.add_subplot(gs[2, 3])
    im12 = ax12.imshow(cl_contribution.numpy(), cmap=cmap_diverging, aspect='auto')
    ax12.set_title('CL: Loss Contribution\n(pred_nonbg * y_at)', fontsize=12, fontweight='bold')
    ax12.set_xlabel('Azimuth bins')
    ax12.set_ylabel('Elevation bins')
    plt.colorbar(im12, ax=ax12, label='Contribution')
    
    # Compute actual loss values for this frame
    intersection = intersection_map.sum()
    union = union_map.sum()
    iou = intersection / (union + 1e-8) if union > 0 else 1.0
    aiur_loss_val = 1.0 - iou
    
    cl_loss_val = cl_contribution.sum() / (N_non * I * J + 1e-8) if N_non > 0 else 0.0
    
    # Overall title
    fig.suptitle(
        f'Loss Component Visualization - Epoch {epoch}, Batch {b_idx}, Frame {t_idx}\n'
        f'AIUR Loss = {aiur_loss_val:.4f} (IoU = {iou:.4f}, Intersection={int(intersection)}, Union={int(union)}) | '
        f'CL Loss = {cl_loss_val:.4f}',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # Save figure
    save_path = os.path.join(save_dir, f'loss_visualization_epoch_{epoch:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Loss visualization saved to {save_path}")
    
    plt.close(fig)
    
    return save_path


# ==============================================================================
# TRAINING FUNCTION (IDENTICAL TO NOTEBOOK + CUDA OPTIMIZATIONS)
# ==============================================================================
def plot_loss_curves(train_losses, test_losses, save_path=None):
    """
    Plot and save training and test loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        test_losses: List of test losses per epoch
        save_path: Path to save the plot (optional)
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
    plt.plot(epochs, test_losses, 'r-s', label='Test Loss', linewidth=2, markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Test Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add minimum loss markers
    min_train_idx = train_losses.index(min(train_losses))
    min_test_idx = test_losses.index(min(test_losses))
    
    plt.plot(min_train_idx + 1, train_losses[min_train_idx], 'b*', 
             markersize=15, label=f'Best Train: {train_losses[min_train_idx]:.4f}')
    plt.plot(min_test_idx + 1, test_losses[min_test_idx], 'r*', 
             markersize=15, label=f'Best Test: {test_losses[min_test_idx]:.4f}')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Loss curve saved to {save_path}")
    
    # Get figure before closing
    fig = plt.gcf()
    
    # Close figure to prevent memory leaks (important for HPC batch jobs)
    plt.close(fig)
    
    return fig


def train_model(
    train_audio_files,
    train_meta_files,
    test_audio_files,
    test_meta_files,
    num_epochs=None,
    batch_size=None,
    learning_rate=None,
    device=None,
    use_small_model=True
):
    """
    Complete training function for SMR-SELD model.
    """
    # Use config defaults if not provided
    num_epochs = config.NUM_EPOCHS
    batch_size = config.BATCH_SIZE
    learning_rate = config.LEARNING_RATE
    device = DEVICE
    
    logger.info("\nStep 1: Loading datasets...")
    
    train_dataset = SELDDataset(
        audio_files=train_audio_files,
        metadata_files=train_meta_files,
        num_classes=config.NUM_CLASSES
    )
    
    test_dataset = SELDDataset(
        audio_files=test_audio_files,
        metadata_files=test_meta_files,
        num_classes=config.NUM_CLASSES
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0,  # Reduced to 2 to save memory
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,  # Reduced to 2 to save memory
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} windows ({len(train_loader)} batches)")
    logger.info(f"Test dataset: {len(test_dataset)} windows ({len(test_loader)} batches)")
    logger.info(f"Grid dimensions: {train_dataset.I}x{train_dataset.J} = {train_dataset.total_cells} cells")
    logger.info("\nStep 2: Initializing model, loss, and optimizer...")
    
    model = SMRSELDWithCSPDarkNet(
        n_channels=config.N_CHANNELS,
        grid_size=(train_dataset.I, train_dataset.J),
        num_classes=config.NUM_CLASSES,
        use_small=use_small_model
    ).to(device)
    
    # Define class weights for CrossEntropyLoss
    # Give higher weight to event classes (0-12) and lower to background (13)
    class_weights = torch.ones(config.NUM_CLASSES).to(device)
    class_weights[config.NUM_CLASSES - 1] = 0.05  # Down-weight background (adjust as needed)
    logger.info(f"Using Class Weights: Events=1.0, Background=0.05")
    
    criterion = SMRSELDLoss(
        w_class=config.W_CLASS,
        w_aiur=config.W_AIUR,
        w_cl=config.W_CL,
        grid_size=(train_dataset.I, train_dataset.J),
        class_weights=class_weights
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler (verbose parameter removed in newer PyTorch versions)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.LR_DECAY_FACTOR,
        patience=config.LR_DECAY_PATIENCE
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Optimizer: Adam (lr={learning_rate}, weight_decay={config.WEIGHT_DECAY})")
    logger.info(f"Scheduler: ReduceLROnPlateau (factor={config.LR_DECAY_FACTOR}, patience={config.LR_DECAY_PATIENCE})")
    
    train_losses = []
    test_losses = []
    best_train_loss = float('inf')
    best_test_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_files = []
    
    logger.info(f"\nTraining configuration:")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Early stopping patience: {config.PATIENCE}")
    logger.info(f"  Min delta: {config.MIN_DELTA}")
    logger.info(f"  Save every N epochs: {config.SAVE_EVERY_N_EPOCHS}")
    logger.info(f"  Device: {device}")
    
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING LOOP")
    logger.info("="*80 + "\n")
    
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = datetime.now()
        model.train()
        train_loss_accum = 0.0
        train_class_ce_accum = 0.0
        train_aiur_accum = 0.0
        train_cl_accum = 0.0
        
        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs} [Train]",
            leave=False
        )
        
        for batch_idx, (spectrograms, labels) in enumerate(train_progress):
            # CUDA optimization: non_blocking transfer for async data movement
            spectrograms = spectrograms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(spectrograms)
            
            # Compute loss
            loss, breakdown = criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            train_loss_accum += loss.item()
            train_class_ce_accum += breakdown['class_ce']
            # train_aiur_accum += breakdown['aiur']
            # train_cl_accum += breakdown['cl']
            
            # Update progress bar
            train_progress.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Average training losses
        avg_train_loss = train_loss_accum / len(train_loader)
        avg_train_class_ce = train_class_ce_accum / len(train_loader)
        # avg_train_aiur = train_aiur_accum / len(train_loader)
        # avg_train_cl = train_cl_accum / len(train_loader)
        
        model.eval()
        test_loss_accum = 0.0
        test_class_ce_accum = 0.0
        # test_aiur_accum = 0.0
        # test_cl_accum = 0.0
        
        test_progress = tqdm(
            test_loader,
            desc=f"Epoch {epoch}/{num_epochs} [Test]",
            leave=False
        )
        
        with torch.no_grad():
            for spectrograms, labels in test_progress:
                # CUDA optimization: non_blocking transfer for async data movement
                spectrograms = spectrograms.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Forward pass
                predictions = model(spectrograms)
                
                # Compute loss
                loss, breakdown = criterion(predictions, labels)
                
                # Accumulate losses
                test_loss_accum += loss.item()
                test_class_ce_accum += breakdown['class_ce']
                # test_aiur_accum += breakdown['aiur']
                # test_cl_accum += breakdown['cl']
                
                # Update progress bar
                test_progress.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Average test losses
        avg_test_loss = test_loss_accum / len(test_loader)
        avg_test_class_ce = test_class_ce_accum / len(test_loader)
        # avg_test_aiur = test_aiur_accum / len(test_loader)
        # avg_test_cl = test_cl_accum / len(test_loader)
        
        # Store losses
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        # Get current learning rate before scheduler step
        old_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate scheduler
        scheduler.step(avg_test_loss)
        
        # Check if learning rate changed
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            logger.info(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Calculate epoch duration
        epoch_duration = (datetime.now() - epoch_start_time).total_seconds()
        
        logger.info(f"\nEpoch {epoch}/{num_epochs} - Duration: {epoch_duration:.1f}s")
        logger.info(f"  Train Loss: {avg_train_loss:.6f} (CE: {avg_train_class_ce:.6f})")
        logger.info(f"  Test Loss:  {avg_test_loss:.6f} (CE: {avg_test_class_ce:.6f})")
        logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping based on train loss
        if avg_train_loss < best_train_loss - config.MIN_DELTA:
            improvement = best_train_loss - avg_train_loss
            best_train_loss = avg_train_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            logger.info(f"  Train loss improved by {improvement:.6f}")
        else:
            epochs_without_improvement += 1
            logger.info(f"  No train loss improvement for {epochs_without_improvement} epoch(s)")
        
        # Save best model based on test loss (for generalization)
        if avg_test_loss < best_test_loss - config.MIN_DELTA:
            improvement = best_test_loss - avg_test_loss
            best_test_loss = avg_test_loss
            
            best_model_path = config.CHECKPOINT_PATH / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'config': config
            }, best_model_path)
            
            logger.info(f"  New best model saved! (test loss improvement: {improvement:.6f})")
        

        if epoch % config.SAVE_EVERY_N_EPOCHS == 0:
            checkpoint_path = config.CHECKPOINT_PATH / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'config': config
            }, checkpoint_path)
            
            checkpoint_files.append(checkpoint_path)
            logger.info(f"  Checkpoint saved: {checkpoint_path.name}")
            
            # Keep only last N checkpoints
            if len(checkpoint_files) > config.KEEP_LAST_N_CHECKPOINTS:
                old_checkpoint = checkpoint_files.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f"  Removed old checkpoint: {old_checkpoint.name}")
        
        # CUDA optimization: Clear CUDA cache periodically to prevent OOM
        if torch.cuda.is_available() and epoch % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"  GPU memory cleared")
        
        # Visualize loss components every 5 epochs
        # if epoch % 5 == 0:
        #     logger.info(f"  Generating loss visualization for epoch {epoch}...")
        #     try:
        #         # Get a batch from test loader for visualization
        #         model.eval()
        #         with torch.no_grad():
        #             for spectrograms, labels in test_loader:
        #                 spectrograms = spectrograms.to(device, non_blocking=True)
        #                 labels = labels.to(device, non_blocking=True)
        #                 predictions = model(spectrograms)
                        
        #                 # Visualize first batch only
        #                 visualize_loss_components(
        #                     predictions, 
        #                     labels, 
        #                     criterion, 
        #                     epoch,
        #                     save_dir=str(config.OUTPUT_PATH / 'train_visualizations')
        #                 )
        #                 break  # Only visualize one batch
        #     except Exception as e:
        #         logger.warning(f"  Could not generate loss visualization: {e}")
        
        if epochs_without_improvement >= config.PATIENCE:
            logger.info(f"\n{'='*80}")
            logger.info(f"EARLY STOPPING at epoch {epoch}")
            logger.info(f"No train loss improvement for {config.PATIENCE} consecutive epochs")
            logger.info(f"Best train loss: {best_train_loss:.6f} at epoch {best_epoch}")
            logger.info(f"Best test loss: {best_test_loss:.6f}")
            logger.info(f"{'='*80}\n")
            break
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total epochs trained: {epoch}")
    logger.info(f"Best train loss: {best_train_loss:.6f} at epoch {best_epoch}")
    logger.info(f"Best test loss: {best_test_loss:.6f}")
    logger.info(f"Final train loss: {train_losses[-1]:.6f}")
    logger.info(f"Final test loss: {test_losses[-1]:.6f}")
    
    logger.info("\nGenerating loss curves...")
    loss_curve_path = config.OUTPUT_PATH / f"loss_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_loss_curves(train_losses, test_losses, save_path=loss_curve_path)
    
    logger.info("\nLoading best model weights...")
    best_checkpoint = safe_torch_load(config.CHECKPOINT_PATH / "best_model.pth")
    model.load_state_dict(best_checkpoint['model_state_dict'])
    logger.info(f"Best model loaded from epoch {best_checkpoint['epoch']}")
    
    history = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'best_train_loss': best_train_loss,
        'best_test_loss': best_test_loss,
        'best_epoch': best_epoch,
        'total_epochs': epoch,
        'config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'grid_size': (train_dataset.I, train_dataset.J)
        }
    }
    
    # Save history to file
    history_path = config.OUTPUT_PATH / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save(history, history_path)
    logger.info(f"Training history saved to {history_path}")
    
    logger.info("\n" + "="*80)
    logger.info("ALL DONE!")
    logger.info("="*80 + "\n")
    
    return model, history


def visualize_grid_predictions(
    ground_truth,
    predictions,
    time_frame,
    grid_size,
    title_prefix="",
    save_path=None
):
    """
    Visualize ground truth and predictions on a 2D grid for a specific time frame.
    """
    I, J = grid_size
    
    # Get class predictions (argmax)
    gt_classes = torch.argmax(ground_truth, dim=-1).cpu().numpy()  # (grid_cells,)
    # Get class predictions (argmax)
    # predictions are logits, so argmax works directly
    pred_classes = torch.argmax(predictions, dim=-1).cpu().numpy()  # (grid_cells,)
    
    # Reshape to 2D grid
    gt_grid = gt_classes.reshape(I, J)
    pred_grid = pred_classes.reshape(I, J)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.cm.get_cmap('tab20', 14)
    
    # Plot ground truth
    im1 = axes[0].imshow(gt_grid, cmap=cmap, vmin=0, vmax=13, aspect='auto')
    axes[0].set_title(f'{title_prefix}Ground Truth\nFrame {time_frame}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Azimuth bins (J)', fontsize=11)
    axes[0].set_ylabel('Elevation bins (I)', fontsize=11)
    axes[0].grid(True, alpha=0.3, color='gray', linewidth=0.5)
    
    # Plot predictions
    im2 = axes[1].imshow(pred_grid, cmap=cmap, vmin=0, vmax=13, aspect='auto')
    axes[1].set_title(f'{title_prefix}Predictions\nFrame {time_frame}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Azimuth bins (J)', fontsize=11)
    axes[1].set_ylabel('Elevation bins (I)', fontsize=11)
    axes[1].grid(True, alpha=0.3, color='gray', linewidth=0.5)
    
    # Plot difference (correct=green, incorrect=red, background=white)
    difference = (gt_classes == pred_classes).astype(int)
    # Mask background cells
    is_background = (gt_classes == 13)
    difference[is_background] = 2  # Special value for background
    
    diff_grid = difference.reshape(I, J)
    diff_cmap = plt.matplotlib.colors.ListedColormap(['red', 'green', 'lightgray'])
    im3 = axes[2].imshow(diff_grid, cmap=diff_cmap, vmin=0, vmax=2, aspect='auto')
    axes[2].set_title(f'{title_prefix}Comparison\nFrame {time_frame}\n(Green=Correct, Red=Wrong, Gray=Background)', 
                      fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Azimuth bins (J)', fontsize=11)
    axes[2].set_ylabel('Elevation bins (I)', fontsize=11)
    axes[2].grid(True, alpha=0.3, color='gray', linewidth=0.5)
    
    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Class ID', fontsize=10)
    
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Class ID', fontsize=10)
    
    # Calculate accuracy (excluding background)
    non_bg_mask = ~is_background
    if non_bg_mask.sum() > 0:
        accuracy = (gt_classes[non_bg_mask] == pred_classes[non_bg_mask]).mean() * 100
        bg_accuracy = (gt_classes[is_background] == pred_classes[is_background]).mean() * 100
    else:
        accuracy = 0.0
        bg_accuracy = (gt_classes == pred_classes).mean() * 100
    
    # Add statistics text
    stats_text = f"Non-BG Accuracy: {accuracy:.1f}%\nBG Accuracy: {bg_accuracy:.1f}%\n"
    stats_text += f"Active Events: {non_bg_mask.sum()}/{len(gt_classes)}"
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    # Close figure to prevent memory leaks (important for HPC batch jobs)
    plt.close(fig)
    
    return fig


def test_model(
    test_audio_files,
    test_meta_files,
    model_path=None,
    batch_size=None,
    device=None,
    num_visualizations=5,
    save_visualizations=True
):
    """
    Test a trained SMR-SELD model and visualize predictions.
    
    This function:
    1. Loads the best model weights
    2. Creates predictions and computes loss on test set
    3. Visualizes ground truth vs predictions for random frames with active events.
    """
    batch_size = batch_size or config.BATCH_SIZE
    device = device or DEVICE
    model_path = model_path or (config.CHECKPOINT_PATH / "best_model.pth")
    
    logger.info("="*80)
    logger.info("STARTING MODEL TESTING")
    logger.info("="*80)
    
    logger.info("\nStep 1: Loading test dataset...")
    
    test_dataset = SELDDataset(
        audio_files=test_audio_files,
        metadata_files=test_meta_files,
        num_classes=config.NUM_CLASSES
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,  # Reduced to 2 to save memory
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info(f"Test dataset: {len(test_dataset)} windows ({len(test_loader)} batches)")
    logger.info(f"Grid dimensions: {test_dataset.I}x{test_dataset.J} = {test_dataset.total_cells} cells")
    
    logger.info(f"\nStep 2: Loading model from {model_path}...")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = safe_torch_load(model_path, map_location=device)
    
    model = SMRSELDWithCSPDarkNet(
        n_channels=config.N_CHANNELS,
        grid_size=(test_dataset.I, test_dataset.J),
        num_classes=config.NUM_CLASSES,
        use_small=True  # Adjust if needed
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded successfully!")
    logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
    logger.info(f"  Checkpoint test loss: {checkpoint['test_loss']:.6f}")
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = SMRSELDLoss(
        w_class=config.W_CLASS,
        w_aiur=config.W_AIUR,
        w_cl=config.W_CL,
        grid_size=(test_dataset.I, test_dataset.J)
    )

    logger.info("\nStep 3: Running inference on test set...")
    
    test_loss_accum = 0.0
    test_class_mse_accum = 0.0
    # test_aiur_accum = 0.0
    # test_cl_accum = 0.0
    
    all_predictions = []
    all_labels = []
    all_window_indices = []
    
    test_progress = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for batch_idx, (spectrograms, labels) in enumerate(test_progress):
            # CUDA optimization: non_blocking transfer for async data movement
            spectrograms = spectrograms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = model(spectrograms)
            loss, breakdown = criterion(predictions, labels)
            
            test_loss_accum += loss.item()
            test_class_mse_accum += breakdown['class_mse']
            # test_aiur_accum += breakdown['aiur']
            # test_cl_accum += breakdown['cl']
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_window_indices.extend(range(batch_idx * batch_size, 
                                           batch_idx * batch_size + spectrograms.shape[0]))
            
            test_progress.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Average test losses
    avg_test_loss = test_loss_accum / len(test_loader)
    avg_test_class_mse = test_class_mse_accum / len(test_loader)
    # avg_test_aiur = test_aiur_accum / len(test_loader)
    # avg_test_cl = test_cl_accum / len(test_loader)
    
    all_predictions = torch.cat(all_predictions, dim=0)  # (N, T, G, M)
    all_labels = torch.cat(all_labels, dim=0)  # (N, T, G, M)
    
    logger.info("\n" + "="*80)
    logger.info("TEST RESULTS")
    logger.info("="*80)
    logger.info(f"Total Loss: {avg_test_loss:.6f}")
    logger.info(f"  Class MSE:  {avg_test_class_mse:.6f}")
    # logger.info(f"  AIUR Loss:  {avg_test_aiur:.6f}")
    # logger.info(f"  CL Loss:    {avg_test_cl:.6f}")
    
    # Calculate overall accuracy
    pred_classes = torch.argmax(all_predictions, dim=-1)
    true_classes = torch.argmax(all_labels, dim=-1)
    overall_accuracy = (pred_classes == true_classes).float().mean().item() * 100
    
    # Calculate non-background accuracy
    is_background = (true_classes == config.NUM_CLASSES - 1)
    non_bg_mask = ~is_background
    if non_bg_mask.sum() > 0:
        non_bg_accuracy = (pred_classes[non_bg_mask] == true_classes[non_bg_mask]).float().mean().item() * 100
    else:
        non_bg_accuracy = 0.0
    
    logger.info(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
    logger.info(f"Non-Background Accuracy: {non_bg_accuracy:.2f}%")
    logger.info(f"Active Events: {non_bg_mask.sum().item()} / {non_bg_mask.numel()}")

    # ========== Loss Component Visualizations ==========
    # logger.info(f"\nStep 4: Generating loss component visualizations...")
    
    # if save_visualizations:
    #     try:
    #         # Find frames with sufficient events for loss visualization
    #         loss_viz_frames = []
    #         for window_idx in range(N):
    #             for time_idx in range(T):
    #                 frame_labels = all_labels[window_idx, time_idx, :, :]
    #                 frame_classes = torch.argmax(frame_labels, dim=-1)
    #                 num_active = (frame_classes != config.NUM_CLASSES - 1).sum().item()
                    
    #                 if num_active >= 5:  # At least 5 active events
    #                     loss_viz_frames.append({
    #                         'window_idx': window_idx,
    #                         'time_idx': time_idx,
    #                         'num_active': num_active
    #                     })
            
    #         if len(loss_viz_frames) > 0:
    #             # Select top 5 frames with most events
    #             loss_viz_frames = sorted(loss_viz_frames, key=lambda x: x['num_active'], reverse=True)
    #             num_loss_viz = min(5, len(loss_viz_frames))
                
    #             logger.info(f"  Generating {num_loss_viz} loss component visualizations...")
                
    #             for viz_idx in range(num_loss_viz):
    #                 frame_info = loss_viz_frames[viz_idx]
    #                 window_idx = frame_info['window_idx']
    #                 time_idx = frame_info['time_idx']
                    
    #                 # Get predictions and labels for this specific frame
    #                 # Need to add batch and time dimensions back for visualize_loss_components
    #                 frame_pred = all_predictions[window_idx:window_idx+1, time_idx:time_idx+1, :, :].to(device)
    #                 frame_label = all_labels[window_idx:window_idx+1, time_idx:time_idx+1, :, :].to(device)
                    
    #                 logger.info(f"    Loss viz {viz_idx + 1}/{num_loss_viz}: Window {window_idx}, Frame {time_idx}, Active cells: {frame_info['num_active']}")
                    
    #                 visualize_loss_components(
    #                     frame_pred,
    #                     frame_label,
    #                     criterion,
    #                     epoch=f"test_viz{viz_idx+1}",  # Use identifier instead of epoch number
    #                     save_dir=str(config.OUTPUT_PATH / 'test_visualizations'),
    #                     frame_idx=0  # Since we're passing single frame
    #                 )
                
    #             logger.info(f"  Loss component visualizations saved to: {config.OUTPUT_PATH / 'test_visualizations'}")
    #         else:
    #             logger.warning("  No frames with >= 5 events found for loss visualization")
        
    #     except Exception as e:
    #         logger.warning(f"  Could not generate loss visualizations: {e}")
    #         import traceback
    #         logger.warning(traceback.format_exc())

    logger.info(f"\nStep 5: Finding frames with active events for prediction visualization...")
    
    # Find frames with non-background events
    N, T, G, M = all_predictions.shape
    frames_with_events = []
    
    for window_idx in range(N):
        for time_idx in range(T):
            # Check if this frame has any non-background events
            frame_labels = all_labels[window_idx, time_idx, :, :]  # (G, M)
            frame_classes = torch.argmax(frame_labels, dim=-1)  # (G,)
            
            # Count non-background cells
            num_active = (frame_classes != config.NUM_CLASSES - 1).sum().item()
            
            if num_active > 0:
                frames_with_events.append({
                    'window_idx': window_idx,
                    'time_idx': time_idx,
                    'num_active': num_active
                })
    
    logger.info(f"Found {len(frames_with_events)} frames with active events")
    
    if len(frames_with_events) == 0:
        logger.warning("No frames with active events found! Cannot create visualizations.")
        return {
            'test_loss': avg_test_loss,
            'class_mse': avg_test_class_mse,
            # 'aiur': avg_test_aiur,
            # 'cl': avg_test_cl,
            'overall_accuracy': overall_accuracy,
            'non_bg_accuracy': non_bg_accuracy,
            'visualizations': []
        }
    
    logger.info(f"\nStep 6: Creating {num_visualizations} prediction visualizations...")
    
    num_to_visualize = min(num_visualizations, len(frames_with_events))
    selected_frames = random.sample(frames_with_events, num_to_visualize)
    
    selected_frames = sorted(selected_frames, key=lambda x: x['num_active'], reverse=True)
    
    visualizations = []
    
    for viz_idx, frame_info in enumerate(selected_frames):
        window_idx = frame_info['window_idx']
        time_idx = frame_info['time_idx']
        num_active = frame_info['num_active']
        
        logger.info(f"\n  Visualization {viz_idx + 1}/{num_to_visualize}:")
        logger.info(f"    Window: {window_idx}, Time Frame: {time_idx}")
        logger.info(f"    Active Events: {num_active}")
        
        frame_predictions = all_predictions[window_idx, time_idx, :, :]  # (G, M)
        frame_labels = all_labels[window_idx, time_idx, :, :]  # (G, M)
        
        save_path = None
        if save_visualizations:
            viz_dir = config.OUTPUT_PATH / "test_visualizations"
            viz_dir.mkdir(exist_ok=True)
            save_path = viz_dir / f"test_viz_{viz_idx + 1}_window{window_idx}_frame{time_idx}.png"
        
        fig = visualize_grid_predictions(
            ground_truth=frame_labels,
            predictions=frame_predictions,
            time_frame=time_idx,
            grid_size=(test_dataset.I, test_dataset.J),
            title_prefix=f"Window {window_idx}, ",
            save_path=save_path
        )
        
        visualizations.append({
            'window_idx': window_idx,
            'time_idx': time_idx,
            'num_active': num_active,
            'figure': fig,
            'save_path': save_path
        })
    
    logger.info("\n" + "="*80)
    logger.info("TESTING COMPLETE")
    logger.info("="*80)

    results = {
        'test_loss': avg_test_loss,
        'class_mse': avg_test_class_mse,
        # 'aiur': avg_test_aiur,
        # 'cl': avg_test_cl,
        'overall_accuracy': overall_accuracy,
        'non_bg_accuracy': non_bg_accuracy,
        'num_frames_with_events': len(frames_with_events),
        'visualizations': visualizations,
        'checkpoint_epoch': checkpoint['epoch']
    }
    
    return results



def main():
    """Main entry point for HPC training"""
    # Initialize logger and config
    global logger, config, DEVICE
    logger, log_file = setup_logging()
    config = Config()
    
    DEVICE = get_device()
    
    logger.info("="*80)
    logger.info("SMR-SELD TRAINING ON HPC")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Configuration:")
    logger.info(f"  Use full dataset: {config.USE_FULL_DATASET}")
    logger.info(f"  Epochs: {config.NUM_EPOCHS}")
    logger.info(f"  Batch size: {config.BATCH_SIZE}")
    logger.info(f"  Learning rate: {config.LEARNING_RATE}")
    
    try:
        # Load files
        logger.info("\nLoading data files...")
        train_audio_files, train_meta_files, test_audio_files, test_meta_files = load_files()
        logger.info(f"Training files: {len(train_audio_files)} audio, {len(train_meta_files)} metadata")
        logger.info(f"Test files: {len(test_audio_files)} audio, {len(test_meta_files)} metadata")
        
        # Train model
        model, history = train_model(
            train_audio_files=train_audio_files,
            train_meta_files=train_meta_files,
            test_audio_files=test_audio_files,
            test_meta_files=test_meta_files,
            num_epochs=None,  # Will use config.NUM_EPOCHS
            batch_size=None,  # Will use config.BATCH_SIZE
            learning_rate=None,  # Will use config.LEARNING_RATE
            device=DEVICE,
            use_small_model=True
        )
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Best model saved to: {config.CHECKPOINT_PATH / 'best_model.pth'}")
        logger.info(f"Best test loss: {history['best_test_loss']:.6f} at epoch {history['best_epoch']}")
        
        # Test the trained model
        logger.info("\n" + "="*80)
        logger.info("STARTING MODEL TESTING")
        logger.info("="*80)
        
        test_results = test_model(
            test_audio_files=test_audio_files,
            test_meta_files=test_meta_files,
            model_path=config.CHECKPOINT_PATH / "best_model.pth",
            batch_size=None,  # Will use config.BATCH_SIZE
            device=DEVICE,
            num_visualizations=10,
            save_visualizations=True
        )
        
        logger.info("\n" + "="*80)
        logger.info("TESTING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Test Loss: {test_results['test_loss']:.6f}")
        logger.info(f"  - Class MSE:  {test_results['class_mse']:.6f}")
        # logger.info(f"  - AIUR Loss:  {test_results['aiur']:.6f}")
        # logger.info(f"  - CL Loss:    {test_results['cl']:.6f}")
        logger.info(f"Overall Accuracy: {test_results['overall_accuracy']:.2f}%")
        logger.info(f"Non-Background Accuracy: {test_results['non_bg_accuracy']:.2f}%")
        logger.info(f"Frames with events: {test_results['num_frames_with_events']}")
        logger.info(f"Visualizations created: {len(test_results['visualizations'])}")
        if len(test_results['visualizations']) > 0:
            viz_dir = test_results['visualizations'][0]['save_path'].parent
            logger.info(f"Visualizations saved to: {viz_dir}")
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Training epochs: {history['total_epochs']}")
        logger.info(f"Best epoch: {history['best_epoch']}")
        logger.info(f"Final test accuracy: {test_results['non_bg_accuracy']:.2f}%")
        
        return 0
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("TRAINING FAILED")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
