import torch
import torchaudio
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
import logging
from torch.utils.data import Dataset

from config import Config
from utils import polar_to_grid

# Initialize logger
logger = logging.getLogger('SMR_SELD')
config = Config()

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
