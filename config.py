from pathlib import Path

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
    MODEL_TYPE = 'conformer' # 'cnn' (original) or 'crnn' (new)
    NUM_CLASSES = 14
    N_CHANNELS = 4
    
    # CRNN Hyperparameters
    CRNN_CNN_CHANNELS = [64, 128, 256, 512]
    CRNN_RNN_HIDDEN = 256
    CRNN_RNN_LAYERS = 2
    CRNN_DROPOUT = 0.3
    
    # Conformer Hyperparameters
    CONF_D_MODEL = 256
    CONF_N_HEADS = 4
    CONF_N_LAYERS = 2
    CONF_KERNEL_SIZE = 31
    CONF_DROPOUT = 0.3
    
    # Training hyperparameters
    NUM_EPOCHS = 30
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    LR_DECAY_FACTOR = 0.5
    LR_DECAY_PATIENCE = 5
    WEIGHT_DECAY = 1e-4
    
    # Loss weights
    LOSS_TYPE = 'ce' # 'ce' (Cross Entropy) or 'mse' (Mean Squared Error)
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
