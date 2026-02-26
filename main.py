#!/usr/bin/env python3
"""
SMR-SELD Training Script
========================

This is the main entry point for training the SMR-SELD model.
It uses the modularized code structure.
"""

import sys
import gc
import logging
import torch
from torch.utils.data import DataLoader

from config import Config
from utils import setup_logging, get_device
from dataset import load_files, SELDDataset
from trainer import train_model, test_model

def main():
    """Main entry point for HPC training"""
    # Config must be created first so that the run directory (and LOG_DIR)
    # exists before setup_logging tries to write into it.
    config = Config()
    logger, log_file = setup_logging(log_dir=str(config.LOG_DIR))
    
    device = get_device(logger)
    
    logger.info("="*80)
    logger.info("SMR-SELD TRAINING ON HPC")
    logger.info("="*80)
    logger.info(f"Log file     : {log_file}")
    logger.info(f"Run directory: {config.RUN_DIR}")

    logger.info("\n" + "─"*80)
    logger.info("DATASET")
    logger.info("─"*80)
    logger.info(f"  Use full dataset   : {config.USE_FULL_DATASET}")
    logger.info(f"  Input features     : {config.INPUT_FEATURES}  →  {config.N_CHANNELS} channels")
    logger.info(f"  Grid cell size     : {config.GRID_CELL_DEGREES}°  →  "
                f"{int(180 // config.GRID_CELL_DEGREES)} × {int(360 // config.GRID_CELL_DEGREES)} grid "
                f"({int(180 // config.GRID_CELL_DEGREES) * int(360 // config.GRID_CELL_DEGREES)} cells)")
    logger.info(f"  Mel bins (n_mels)  : {config.N_MELS}")
    logger.info(f"  FFT size           : {config.SPECTROGRAM_N_FFT}  ({config.SPECTROGRAM_N_FFT / config.SR * 1000:.0f} ms)")
    logger.info(f"  Hop length         : {config.SPECTROGRAM_HOP_LENGTH}  ({config.SPECTROGRAM_HOP_LENGTH / config.SR * 1000:.0f} ms per frame)")
    logger.info(f"  Window length      : {config.WINDOW_LENGTH} samples  ({config.WINDOW_LENGTH / config.SR:.0f} s)")
    logger.info(f"  Window hop         : {config.HOP_LENGTH} samples  ({config.HOP_LENGTH / config.SR:.0f} s)"    )
    logger.info(f"  Sample rate        : {config.SR} Hz")
    logger.info(f"  Num classes        : {config.NUM_CLASSES}  (including background)")

    logger.info("\n" + "─"*80)
    logger.info("MODEL")
    logger.info("─"*80)
    logger.info(f"  Model type         : {config.MODEL_TYPE}")
    if config.MODEL_TYPE == 'crnn':
        logger.info(f"  CNN channels       : {config.CRNN_CNN_CHANNELS}")
        logger.info(f"  RNN hidden size    : {config.CRNN_RNN_HIDDEN}")
        logger.info(f"  RNN layers         : {config.CRNN_RNN_LAYERS}")
        logger.info(f"  Dropout            : {config.CRNN_DROPOUT}")
    elif config.MODEL_TYPE == 'conformer':
        logger.info(f"  d_model            : {config.CONF_D_MODEL}")
        logger.info(f"  Attention heads    : {config.CONF_N_HEADS}")
        logger.info(f"  Conformer layers   : {config.CONF_N_LAYERS}")
        logger.info(f"  Conv kernel size   : {config.CONF_KERNEL_SIZE}")
        logger.info(f"  Dropout            : {config.CONF_DROPOUT}")
    elif config.MODEL_TYPE == 'resnet_conformer':
        logger.info(f"  d_model            : {config.RESNET_CONF_D_MODEL}")
        logger.info(f"  Attention heads    : {config.RESNET_CONF_N_HEADS}")
        logger.info(f"  Conformer layers   : {config.RESNET_CONF_N_LAYERS}")
        logger.info(f"  Dropout            : {config.RESNET_DROPOUT}")

    logger.info("\n" + "─"*80)
    logger.info("TRAINING")
    logger.info("─"*80)
    logger.info(f"  Epochs             : {config.NUM_EPOCHS}")
    logger.info(f"  Batch size         : {config.BATCH_SIZE}")
    logger.info(f"  Learning rate      : {config.LEARNING_RATE}")
    logger.info(f"  Weight decay       : {config.WEIGHT_DECAY}")
    logger.info(f"  LR decay factor    : {config.LR_DECAY_FACTOR}")
    logger.info(f"  LR decay patience  : {config.LR_DECAY_PATIENCE} epochs")
    logger.info(f"  Early stop patience: {config.PATIENCE} epochs")
    logger.info(f"  Min delta          : {config.MIN_DELTA}")
    logger.info(f"  Save every N epochs: {config.SAVE_EVERY_N_EPOCHS}")
    logger.info(f"  Keep last N ckpts  : {config.KEEP_LAST_N_CHECKPOINTS}")
    logger.info(f"  Visualize every N  : {config.VISUALIZE_EVERY_N_EPOCHS} epochs")

    logger.info("\n" + "─"*80)
    logger.info("LOSS")
    logger.info("─"*80)
    logger.info(f"  Loss type          : {config.LOSS_TYPE.upper()}")
    logger.info(f"  w_class            : {config.W_CLASS}")
    logger.info(f"  w_aiur             : {config.W_AIUR}  (currently inactive)")
    logger.info(f"  w_cl               : {config.W_CL}   (currently inactive)")
    logger.info(f"  w_trunc_l1         : {config.W_TRUNC_L1}"
                + ("  (DISABLED)" if config.W_TRUNC_L1 == 0 else "  (ENABLED)"))
    if config.W_TRUNC_L1 > 0:
        logger.info(f"    trunc_thresh_class  : {config.TRUNC_THRESH_CLASS}")
        logger.info(f"    trunc_thresh_spatial: {config.TRUNC_THRESH_SPATIAL}")
    logger.info(f"  Masked loss        : {config.USE_MASKED_LOSS}")
    if config.USE_MASKED_LOSS:
        logger.info(f"    bg keep ratio   : {config.MASK_BG_RATIO}× positives")

    logger.info("\n" + "─"*80)
    logger.info(f"Device             : {device}")
    logger.info("─"*80 + "\n")
    
    try:
        # Load files
        logger.info("\nLoading data files...")
        train_audio_files, train_meta_files, test_audio_files, test_meta_files = load_files()
        logger.info(f"Training files: {len(train_audio_files)} audio, {len(train_meta_files)} metadata")
        logger.info(f"Test files: {len(test_audio_files)} audio, {len(test_meta_files)} metadata")
        
        # Create Datasets and DataLoaders
        logger.info("Initializing Datasets and DataLoaders...")
        
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
        
        # Log statistics
        logger.info("\n" + "="*40)
        logger.info("TRAINING SET STATISTICS")
        logger.info("="*40)
        train_dataset.get_statistics()
        
        logger.info("\n" + "="*40)
        logger.info("TEST SET STATISTICS")
        logger.info("="*40)
        test_dataset.get_statistics()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=2 if torch.cuda.is_available() else 0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=2 if torch.cuda.is_available() else 0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Train model
        model, history = train_model(
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=config.NUM_EPOCHS,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            device=device
        )
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Best model saved to: {config.CHECKPOINT_PATH / 'best_model.pth'}")
        logger.info(f"Best test loss: {history['best_test_loss']:.6f} at epoch {history['best_epoch']}")

        # Free the training model – test_model loads its own fresh copy from
        # the checkpoint, so keeping this object alive wastes GPU memory.
        logger.info("Releasing training model from memory before inference...")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"GPU memory after cleanup: "
                        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, "
                        f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
        logger.info("\n" + "="*80)
        logger.info("STARTING MODEL TESTING")
        logger.info("="*80)
        
        test_results = test_model(
            test_loader=test_loader,
            model_path=config.CHECKPOINT_PATH / "best_model.pth",
            batch_size=config.BATCH_SIZE,
            device=device,
            num_visualizations=10,
            save_visualizations=True
        )
        
        logger.info("\n" + "="*80)
        logger.info("TESTING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Test Loss: {test_results['test_loss']:.6f}")
        logger.info(f"  - Class {config.LOSS_TYPE.upper()}:  {test_results[f'class_{config.LOSS_TYPE}']:.6f}")
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
