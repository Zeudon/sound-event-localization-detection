import os
import sys
import logging
import torch
import numpy as np
from datetime import datetime

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

def get_device(logger=None):
    """Get available device with CUDA support"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if logger:
            logger.info(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # CUDA optimization: Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        if logger:
            logger.info("CuDNN benchmarking enabled for optimized performance")
    else:
        device = torch.device('cpu')
        if logger:
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
