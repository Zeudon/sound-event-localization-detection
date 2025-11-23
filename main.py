#!/usr/bin/env python3
"""
SMR-SELD Training Script
========================

This is the main entry point for training the SMR-SELD model.
It uses the modularized code structure.
"""

import sys
import logging
import torch
from torch.utils.data import DataLoader

from config import Config
from utils import setup_logging, get_device
from dataset import load_files, SELDDataset
from trainer import train_model, test_model

def main():
    """Main entry point for HPC training"""
    # Initialize logger and config
    logger, log_file = setup_logging()
    config = Config()
    
    device = get_device(logger)
    
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
        
        # Test the trained model
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
