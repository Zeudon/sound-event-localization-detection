import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt

logger = logging.getLogger('SMR_SELD')

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
