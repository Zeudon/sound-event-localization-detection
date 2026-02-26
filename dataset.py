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

def audio_to_mag_ipd_features(waveform, sample_rate, n_fft=None, hop_length=None, n_mels=None):
    """
    Compute log-mel magnitude + Inter-channel Phase Difference (IPD) features.

    For FOA audio (channels W, X, Y, Z) this produces 10 feature channels:
      - 4  log-mel magnitude channels  (W, X, Y, Z independently)
      - 3  cos(IPD) channels           (W→X, W→Y, W→Z)
      - 3  sin(IPD) channels           (W→X, W→Y, W→Z)

    IPD between reference channel W (ch0) and channel i:
        IPD(t,f) = angle( STFT_W(t,f) * conj(STFT_i(t,f)) )
    Encoded as cos+sin to avoid phase-wrap discontinuities.
    The mel filterbank is applied to cos/sin in the frequency domain
    before returning, so all 10 channels share the same (n_mels, T) shape.

    Returns:
        features: (10, n_mels, T)
    """
    if n_fft is None:
        n_fft = config.SPECTROGRAM_N_FFT
    if hop_length is None:
        hop_length = config.SPECTROGRAM_HOP_LENGTH
    if n_mels is None:
        n_mels = config.N_MELS

    n_freqs = n_fft // 2 + 1
    window = torch.hann_window(n_fft, device=waveform.device)

    # ── Step 1: complex STFT for all 4 channels ─────────────────────────
    stfts = []
    for ch in range(waveform.shape[0]):
        stft = torch.stft(
            waveform[ch],
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )  # (n_freqs, T)
        stfts.append(stft)

    # ── Step 2: mel filterbank matrix (n_mels, n_freqs) ──────────────────
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_freqs,
        f_min=0.0,
        f_max=float(sample_rate) / 2.0,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm=None,
        mel_scale='htk'
    ).T  # (n_mels, n_freqs)

    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    # ── Step 3: log-mel magnitude per channel ────────────────────────────
    mag_features = []
    for ch in range(4):
        magnitude = stfts[ch].abs()              # (n_freqs, T)
        mel_mag = mel_fb @ magnitude             # (n_mels, T)
        # AmplitudeToDB expects (C, F, T) or (F, T); add/remove batch dim
        mel_mag_db = amp_to_db(mel_mag.unsqueeze(0)).squeeze(0)  # (n_mels, T)
        mag_features.append(mel_mag_db)

    # ── Step 4: cos/sin IPD between W (ch0) and X,Y,Z (ch1,ch2,ch3) ─────
    ipd_features = []
    for ch in range(1, 4):
        cross = stfts[0] * stfts[ch].conj()     # (n_freqs, T)  complex
        denom = cross.abs() + 1e-8
        cos_ipd = cross.real / denom             # (n_freqs, T)
        sin_ipd = cross.imag / denom             # (n_freqs, T)

        mel_cos = mel_fb @ cos_ipd               # (n_mels, T)
        mel_sin = mel_fb @ sin_ipd               # (n_mels, T)
        ipd_features.append(mel_cos)
        ipd_features.append(mel_sin)

    # ── Step 5: concatenate → (10, n_mels, T) ────────────────────────────
    features = torch.stack(mag_features + ipd_features, dim=0)
    return features


def audio_to_mag_phase_features(waveform, sample_rate, n_fft=None, hop_length=None, n_mels=None):
    """
    Compute log-mel magnitude + per-channel mel-phase features.

    For FOA audio (channels W, X, Y, Z) this produces 12 feature channels:
      - 4  log-mel magnitude channels          (W, X, Y, Z)
      - 4  cos(mel-phase) channels             (W, X, Y, Z)
      - 4  sin(mel-phase) channels             (W, X, Y, Z)

    Phase is extracted per channel as the angle of the complex STFT bin,
    then cos/sin encoded (to avoid wrap discontinuities) before applying
    the mel filterbank, so all 12 channels share shape (n_mels, T).

    Unlike IPD, this retains absolute per-channel phase information which
    can help the model learn source characteristics and reverb patterns
    in addition to spatial cues.

    Returns:
        features: (12, n_mels, T)
    """
    if n_fft is None:
        n_fft = config.SPECTROGRAM_N_FFT
    if hop_length is None:
        hop_length = config.SPECTROGRAM_HOP_LENGTH
    if n_mels is None:
        n_mels = config.N_MELS

    n_freqs = n_fft // 2 + 1
    window = torch.hann_window(n_fft, device=waveform.device)

    # ── Step 1: complex STFT for all 4 channels ──────────────────────────
    stfts = []
    for ch in range(waveform.shape[0]):
        stft = torch.stft(
            waveform[ch],
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )  # (n_freqs, T)
        stfts.append(stft)

    # ── Step 2: mel filterbank matrix (n_mels, n_freqs) ──────────────────
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_freqs,
        f_min=0.0,
        f_max=float(sample_rate) / 2.0,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm=None,
        mel_scale='htk'
    ).T  # (n_mels, n_freqs)

    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    # ── Step 3: log-mel magnitude + cos/sin mel-phase per channel ────────
    mag_features   = []
    cos_ph_features = []
    sin_ph_features = []

    for ch in range(4):
        s = stfts[ch]                            # (n_freqs, T) complex

        # Magnitude
        magnitude = s.abs()                      # (n_freqs, T)
        mel_mag = mel_fb @ magnitude             # (n_mels, T)
        mel_mag_db = amp_to_db(mel_mag.unsqueeze(0)).squeeze(0)  # (n_mels, T)
        mag_features.append(mel_mag_db)

        # Phase → cos/sin to avoid wrap discontinuity
        denom = magnitude + 1e-8
        cos_ph = s.real / denom                  # (n_freqs, T)
        sin_ph = s.imag / denom                  # (n_freqs, T)
        cos_ph_features.append(mel_fb @ cos_ph)  # (n_mels, T)
        sin_ph_features.append(mel_fb @ sin_ph)  # (n_mels, T)

    # ── Step 4: concatenate → (12, n_mels, T) ────────────────────────────
    features = torch.stack(mag_features + cos_ph_features + sin_ph_features, dim=0)
    return features


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
                feat_mode = config.INPUT_FEATURES
                if feat_mode == 'mag_ipd':
                    mel_spec = audio_to_mag_ipd_features(
                        waveform, sr,
                        n_fft=self.n_fft,
                        hop_length=self.spectrogram_hop_length,
                        n_mels=self.n_mels
                    )  # Shape: (10, n_mels, time_frames)
                elif feat_mode == 'mag_phase':
                    mel_spec = audio_to_mag_phase_features(
                        waveform, sr,
                        n_fft=self.n_fft,
                        hop_length=self.spectrogram_hop_length,
                        n_mels=self.n_mels
                    )  # Shape: (12, n_mels, time_frames)
                else:  # 'mag'
                    mel_spec = audio_to_mel_spectrogram(
                        waveform, sr,
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

                # Filter out frames that contain no sound events (background-only frames).
                # A frame has no events when every grid cell has only the background class
                # active, i.e. the sum over all non-background class channels is zero.
                # labels shape: (T, I*J, num_classes)  -- background is the last class.
                event_mask = labels[:, :, :-1].sum(dim=(1, 2)) > 0  # (T,) bool
                n_total = event_mask.shape[0]
                n_kept = int(event_mask.sum().item())
                if n_kept < n_total:
                    mel_spec = mel_spec[:, :, event_mask]   # (4, n_mels, T_kept)
                    labels   = labels[event_mask, :, :]     # (T_kept, I*J, num_classes)
                    logger.info(
                        f"  File {idx}: removed {n_total - n_kept} background-only frames "
                        f"({n_kept}/{n_total} frames retained)"
                    )

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

        # ── Cell-level statistics ────────────────────────────────────────────
        # A (frame, cell) pair is "active" when at least one non-background
        # class channel is 1.  Background is the last class channel (index -1).
        # active_cell_mask: (T, I*J) bool
        active_cell_mask = self.concatenated_labels[:, :, :-1].sum(dim=2) > 0
        total_cell_slots  = self.total_frames * self.total_cells   # T × (I*J)
        total_active_cells = int(active_cell_mask.sum().item())
        total_bg_cells     = total_cell_slots - total_active_cells

        logger.info("Cell masking statistics (summed across all frames):")
        logger.info(f"  Total (frame × cell) slots : {total_cell_slots:,}")
        logger.info(f"  Slots with active events   : {total_active_cells:,}  "
                    f"({100.0 * total_active_cells / total_cell_slots:.2f}%)")
        logger.info(f"  Slots with background only : {total_bg_cells:,}  "
                    f"({100.0 * total_bg_cells / total_cell_slots:.2f}%)")

        print("=" * 60)
        print("Cell masking statistics (summed across all frames):")
        print(f"  Total (frame × cell) slots : {total_cell_slots:,}")
        print(f"  Slots with active events   : {total_active_cells:,}  "
              f"({100.0 * total_active_cells / total_cell_slots:.2f}%)")
        print(f"  Slots with background only : {total_bg_cells:,}  "
              f"({100.0 * total_bg_cells / total_cell_slots:.2f}%)")
        print("=" * 60)
    
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
                
                # Pad spectrograms: (C, n_mels, time) -> pad time dimension
                n_feat_channels = window_spec.shape[0]
                spec_pad = torch.zeros((n_feat_channels, self.n_mels, pad_frames), dtype=window_spec.dtype)
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

    def get_statistics(self):
        """
        Calculate statistics about the number of events per frame.
        Returns:
            stats: Dictionary mapping number of events (0, 1, 2, ...) to count of frames
        """
        logger.info("Calculating dataset statistics...")
        
        # We can use self.concatenated_labels which has shape (Total_Frames, Grid_Size, Num_Classes)
        # Background class is the last one (index 13)
        
        # Get class indices for each cell: (Total_Frames, Grid_Size)
        # We want to count how many cells have a non-background class
        
        # Reshape to combine grid cells: (Total_Frames, Grid_Size, Num_Classes)
        labels = self.concatenated_labels
        
        # Check if background class is active (it should be mutually exclusive with events in our setup, 
        # but let's count explicitly non-background events)
        # Class index for background is self.num_classes - 1
        bg_class_idx = self.num_classes - 1
        
        # Get the class index for each cell
        # Since labels are one-hot (or multi-hot if multiple events in same cell? usually 1 event per cell in this encoding)
        # But actually our encoding puts 1.0 for the active class.
        
        # Let's look at how labels are created in metadata_to_labels:
        # labels[t, cell_idx, active_class] = 1.0
        # labels[t, cell_idx, num_classes - 1] = 1.0 (if no event)
        
        # So we can just sum the non-background channels across all grid cells
        # Slice out the non-background classes: [:, :, 0:bg_class_idx]
        event_labels = labels[:, :, :bg_class_idx]
        
        # Sum across grid cells and classes to get total events per frame
        # Shape: (Total_Frames,)
        events_per_frame = event_labels.sum(dim=(1, 2))
        
        # Convert to numpy for easier counting
        events_per_frame_np = events_per_frame.numpy()
        
        # Count occurrences
        unique, counts = np.unique(events_per_frame_np, return_counts=True)
        stats = dict(zip(unique.astype(int), counts))
        
        total_frames = len(events_per_frame_np)
        
        logger.info(f"Dataset Statistics (Total frames: {total_frames}):")
        for num_events in sorted(stats.keys()):
            count = stats[num_events]
            percentage = (count / total_frames) * 100
            logger.info(f"  Frames with {num_events} events: {count} ({percentage:.2f}%)")
            
        return stats
