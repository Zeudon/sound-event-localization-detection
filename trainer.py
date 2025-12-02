import torch
import logging
import gc
import random
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

from config import Config
from dataset import SELDDataset
from model import SMRSELDWithCSPDarkNet
from model_crnn import SELD_CRNN
from model_conformer import SELD_Conformer
from resnet50_model import SELD_ResNet50_Conformer
from loss import SMRSELDLoss
from utils import safe_torch_load
from visualization import plot_loss_curves, visualize_loss_components, visualize_grid_predictions

logger = logging.getLogger('SMR_SELD')
config = Config()

def train_model(
    train_loader,
    test_loader,
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
    
    logger.info("\nStep 1: Using pre-initialized DataLoaders...")
    
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    
    logger.info(f"Train dataset: {len(train_dataset)} windows ({len(train_loader)} batches)")
    logger.info(f"Test dataset: {len(test_dataset)} windows ({len(test_loader)} batches)")
    logger.info(f"Grid dimensions: {train_dataset.I}x{train_dataset.J} = {train_dataset.total_cells} cells")
    logger.info("\nStep 2: Initializing model, loss, and optimizer...")
    
    if config.MODEL_TYPE == 'crnn':
        logger.info("Initializing CRNN model...")
        model = SELD_CRNN(
            n_channels=config.N_CHANNELS,
            n_mels=config.N_MELS,
            grid_size=(train_dataset.I, train_dataset.J),
            num_classes=config.NUM_CLASSES,
            cnn_channels=config.CRNN_CNN_CHANNELS,
            rnn_hidden=config.CRNN_RNN_HIDDEN,
            rnn_layers=config.CRNN_RNN_LAYERS,
            dropout=config.CRNN_DROPOUT
        ).to(device)
    elif config.MODEL_TYPE == 'conformer':
        logger.info("Initializing Conformer model...")
        model = SELD_Conformer(
            n_channels=config.N_CHANNELS,
            n_mels=config.N_MELS,
            grid_size=(train_dataset.I, train_dataset.J),
            num_classes=config.NUM_CLASSES,
            cnn_channels=config.CRNN_CNN_CHANNELS,
            conf_d_model=config.CONF_D_MODEL,
            conf_n_heads=config.CONF_N_HEADS,
            conf_n_layers=config.CONF_N_LAYERS,
            conf_kernel_size=config.CONF_KERNEL_SIZE,
            dropout=config.CONF_DROPOUT
        ).to(device)
    elif config.MODEL_TYPE == 'resnet_conformer':
        logger.info("Initializing ResNet50-Conformer model...")
        model = SELD_ResNet50_Conformer(
            n_channels=config.N_CHANNELS,
            n_mels=config.N_MELS,
            grid_size=(train_dataset.I, train_dataset.J),
            num_classes=config.NUM_CLASSES,
            conf_d_model=config.RESNET_CONF_D_MODEL,
            conf_n_heads=config.RESNET_CONF_N_HEADS,
            conf_n_layers=config.RESNET_CONF_N_LAYERS,
            dropout=config.RESNET_DROPOUT
        ).to(device)
    else:
        logger.info("Initializing CSPDarkNet (CNN) model...")
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
        loss_type=config.LOSS_TYPE,
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
        train_class_loss_accum = 0.0
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
            train_class_loss_accum += breakdown[f'class_{config.LOSS_TYPE}']
            # train_aiur_accum += breakdown['aiur']
            # train_cl_accum += breakdown['cl']
            
            # Update progress bar
            train_progress.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Average training losses
        avg_train_loss = train_loss_accum / len(train_loader)
        avg_train_class_loss = train_class_loss_accum / len(train_loader)
        # avg_train_aiur = train_aiur_accum / len(train_loader)
        # avg_train_cl = train_cl_accum / len(train_loader)
        
        model.eval()
        test_loss_accum = 0.0
        test_class_loss_accum = 0.0
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
                test_class_loss_accum += breakdown[f'class_{config.LOSS_TYPE}']
                # test_aiur_accum += breakdown['aiur']
                # test_cl_accum += breakdown['cl']
                
                # Update progress bar
                test_progress.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Average test losses
        avg_test_loss = test_loss_accum / len(test_loader)
        avg_test_class_loss = test_class_loss_accum / len(test_loader)
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
        logger.info(f"  Train Loss: {avg_train_loss:.6f} ({config.LOSS_TYPE.upper()}: {avg_train_class_loss:.6f})")
        logger.info(f"  Test Loss:  {avg_test_loss:.6f} ({config.LOSS_TYPE.upper()}: {avg_test_class_loss:.6f})")
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
        #                 
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

def test_model(
    test_loader,
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
    model_path = model_path or (config.CHECKPOINT_PATH / "best_model.pth")
    
    logger.info("="*80)
    logger.info("STARTING MODEL TESTING")
    logger.info("="*80)
    
    logger.info("\nStep 1: Using pre-initialized Test DataLoader...")
    test_dataset = test_loader.dataset
    
    logger.info(f"Test dataset: {len(test_dataset)} windows ({len(test_loader)} batches)")
    logger.info(f"Grid dimensions: {test_dataset.I}x{test_dataset.J} = {test_dataset.total_cells} cells")
    
    logger.info(f"\nStep 2: Loading model from {model_path}...")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = safe_torch_load(model_path, map_location=device)
    
    # Determine model type from config or checkpoint
    # Ideally checkpoint stores config, but for now we use global config
    if config.MODEL_TYPE == 'crnn':
        model = SELD_CRNN(
            n_channels=config.N_CHANNELS,
            n_mels=config.N_MELS,
            grid_size=(test_dataset.I, test_dataset.J),
            num_classes=config.NUM_CLASSES,
            cnn_channels=config.CRNN_CNN_CHANNELS,
            rnn_hidden=config.CRNN_RNN_HIDDEN,
            rnn_layers=config.CRNN_RNN_LAYERS,
            dropout=config.CRNN_DROPOUT
        ).to(device)
    elif config.MODEL_TYPE == 'conformer':
        model = SELD_Conformer(
            n_channels=config.N_CHANNELS,
            n_mels=config.N_MELS,
            grid_size=(test_dataset.I, test_dataset.J),
            num_classes=config.NUM_CLASSES,
            cnn_channels=config.CRNN_CNN_CHANNELS,
            conf_d_model=config.CONF_D_MODEL,
            conf_n_heads=config.CONF_N_HEADS,
            conf_n_layers=config.CONF_N_LAYERS,
            conf_kernel_size=config.CONF_KERNEL_SIZE,
            dropout=config.CONF_DROPOUT
        ).to(device)
    elif config.MODEL_TYPE == 'resnet_conformer':
        model = SELD_ResNet50_Conformer(
            n_channels=config.N_CHANNELS,
            n_mels=config.N_MELS,
            grid_size=(test_dataset.I, test_dataset.J),
            num_classes=config.NUM_CLASSES,
            conf_d_model=config.RESNET_CONF_D_MODEL,
            conf_n_heads=config.RESNET_CONF_N_HEADS,
            conf_n_layers=config.RESNET_CONF_N_LAYERS,
            dropout=config.RESNET_DROPOUT
        ).to(device)
    else:
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
        loss_type=config.LOSS_TYPE,
        w_class=config.W_CLASS,
        w_aiur=config.W_AIUR,
        w_cl=config.W_CL,
        grid_size=(test_dataset.I, test_dataset.J)
    )

    logger.info("\nStep 3: Running inference on test set...")
    
    test_loss_accum = 0.0
    test_class_loss_accum = 0.0
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
            test_class_loss_accum += breakdown[f'class_{config.LOSS_TYPE}']
            # test_aiur_accum += breakdown['aiur']
            # test_cl_accum += breakdown['cl']
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_window_indices.extend(range(batch_idx * batch_size, 
                                           batch_idx * batch_size + spectrograms.shape[0]))
            
            test_progress.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Average test losses
    avg_test_loss = test_loss_accum / len(test_loader)
    avg_test_class_loss = test_class_loss_accum / len(test_loader)
    # avg_test_aiur = test_aiur_accum / len(test_loader)
    # avg_test_cl = test_cl_accum / len(test_loader)
    
    all_predictions = torch.cat(all_predictions, dim=0)  # (N, T, G, M)
    all_labels = torch.cat(all_labels, dim=0)  # (N, T, G, M)
    
    logger.info("\n" + "="*80)
    logger.info("TEST RESULTS")
    logger.info("="*80)
    logger.info(f"Total Loss: {avg_test_loss:.6f}")
    logger.info(f"  Class {config.LOSS_TYPE.upper()}:  {avg_test_class_loss:.6f}")
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
    #                 
    #                 if num_active >= 5:  # At least 5 active events
    #                     loss_viz_frames.append({
    #                         'window_idx': window_idx,
    #                         'time_idx': time_idx,
    #                         'num_active': num_active
    #                     })
    #         
    #         if len(loss_viz_frames) > 0:
    #             # Select top 5 frames with most events
    #             loss_viz_frames = sorted(loss_viz_frames, key=lambda x: x['num_active'], reverse=True)
    #             num_loss_viz = min(5, len(loss_viz_frames))
    #             
    #             logger.info(f"  Generating {num_loss_viz} loss component visualizations...")
    #             
    #             for viz_idx in range(num_loss_viz):
    #                 frame_info = loss_viz_frames[viz_idx]
    #                 window_idx = frame_info['window_idx']
    #                 time_idx = frame_info['time_idx']
    #                 
    #                 # Get predictions and labels for this specific frame
    #                 # Need to add batch and time dimensions back for visualize_loss_components
    #                 frame_pred = all_predictions[window_idx:window_idx+1, time_idx:time_idx+1, :, :].to(device)
    #                 frame_label = all_labels[window_idx:window_idx+1, time_idx:time_idx+1, :, :].to(device)
    #                 
    #                 logger.info(f"    Loss viz {viz_idx + 1}/{num_loss_viz}: Window {window_idx}, Frame {time_idx}, Active cells: {frame_info['num_active']}")
    #                 
    #                 visualize_loss_components(
    #                     frame_pred,
    #                     frame_label,
    #                     criterion,
    #                     epoch=f"test_viz{viz_idx+1}",  # Use identifier instead of epoch number
    #                     save_dir=str(config.OUTPUT_PATH / 'test_visualizations'),
    #                     frame_idx=0  # Since we're passing single frame
    #                 )
    #             
    #             logger.info(f"  Loss component visualizations saved to: {config.OUTPUT_PATH / 'test_visualizations'}")
    #         else:
    #             logger.warning("  No frames with >= 5 events found for loss visualization")
    #     
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
        f'class_{config.LOSS_TYPE}': avg_test_class_loss,
        # 'aiur': avg_test_aiur,
        # 'cl': avg_test_cl,
        'overall_accuracy': overall_accuracy,
        'non_bg_accuracy': non_bg_accuracy,
        'num_frames_with_events': len(frames_with_events),
        'visualizations': visualizations,
        'checkpoint_epoch': checkpoint['epoch']
    }
    
    return results
