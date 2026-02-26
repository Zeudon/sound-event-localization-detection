import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SMRSELDLoss(nn.Module):
    """Complete SMR-SELD loss function with three components"""

    def __init__(self, loss_type='ce', w_class=1.0, w_aiur=0.5, w_cl=0.5, 
                 w_trunc_l1=0.0, trunc_thresh_class=0.2, trunc_thresh_spatial=0.2,
                 use_masked_loss=False, mask_bg_ratio=3.0,
                 grid_size=None, class_weights=None):
        super().__init__()
        self.loss_type = loss_type
        self.w_class = w_class
        self.w_aiur = w_aiur
        self.w_cl = w_cl
        self.w_trunc_l1 = w_trunc_l1
        self.trunc_thresh_class = trunc_thresh_class
        self.trunc_thresh_spatial = trunc_thresh_spatial
        self.use_masked_loss = use_masked_loss
        self.mask_bg_ratio = mask_bg_ratio
        
        self.eps = 1e-10  # Small epsilon for numerical stability
        if grid_size is not None:
            self.I, self.J = grid_size
        else:
            self.I = self.J = None
            
        # Initialize CrossEntropyLoss with weights if provided
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        else:
            self.ce_loss = nn.CrossEntropyLoss(reduction='none')
            
    def truncated_l1_loss(self, y_pred):
        """
        Truncated L1 regularization loss.
        y_pred: (B, T, G, M) - Logits

        Goal: promote spatially sparse event predictions without reinforcing
        the background-collapse problem.

        Component 1 – CLASS sparsity (per active cell):
            Penalise the BACKGROUND probability when it is too high, i.e.
            push the model away from assigning near-1.0 background confidence
            to every cell.  Penalty = min(p_bg, trunc_thresh_class).
            This creates a gentle gradient that discourages total background
            collapse while staying bounded (no exploding gradients).

        Component 2 – SPATIAL sparsity (per frame):
            Penalise frames where the total predicted event-activity
            (sum of non-background probs across all cells) exceeds a
            threshold.  This is the original intent – few active cells –
            but applied only *above* the threshold so it never rewards
            predicting background everywhere.
            Penalty = max(0, cell_activity_total - trunc_thresh_spatial).
        """
        probs = F.softmax(y_pred, dim=-1)  # (B, T, G, M)

        # 1. Class sparsity: penalise over-confident background predictions
        #    p_bg close to 1.0 means the model is collapsing → push back.
        p_bg = probs[..., -1]  # (B, T, G)
        trunc_class = torch.min(
            p_bg,
            torch.tensor(self.trunc_thresh_class, device=p_bg.device)
        )
        loss_sparsity_class = trunc_class.mean()

        # 2. Spatial sparsity: penalise excess event activity per frame
        #    (too many cells active at once), but never reward zero activity.
        probs_events = probs[..., :-1]  # (B, T, G, M-1)
        cell_activity = probs_events.sum(dim=-1)  # (B, T, G)  – per-cell activity
        total_activity_per_frame = cell_activity.sum(dim=-1)   # (B, T)

        excess = torch.clamp(
            total_activity_per_frame - self.trunc_thresh_spatial,
            min=0.0
        )
        loss_sparsity_spatial = excess.mean()

        return loss_sparsity_class + loss_sparsity_spatial

    def get_mask_indices(self, y_true):
        """
        Generate mask to sample all positives and ratio*positives negatives
        y_true: (B, T, G, M) one-hot
        """
        B, T, G, M = y_true.shape
        background_idx = M - 1
        
        y_true_indices = torch.argmax(y_true, dim=-1) # (B, T, G)
        
        # Positive mask: where class is NOT background
        pos_mask = (y_true_indices != background_idx) # (B, T, G)
        
        # Negative mask: where class IS background
        neg_mask = (y_true_indices == background_idx) # (B, T, G)
        
        # Flat indices
        pos_indices = torch.nonzero(pos_mask.view(-1), as_tuple=True)[0]
        neg_indices = torch.nonzero(neg_mask.view(-1), as_tuple=True)[0]
        
        num_pos = pos_indices.numel()
        num_neg = neg_indices.numel()
        
        # Determine how many negatives to keep
        if num_pos > 0:
            num_keep_neg = int(num_pos * self.mask_bg_ratio)
            num_keep_neg = min(num_neg, num_keep_neg)
        else:
            # If no positives, maybe keep a small fraction of negatives or just 0?
            # Let's keep a small fixed amount to learn pure background?
            # Or just keep 0? If 0, loss is 0 for this batch.
            # Let's keep 10% of negatives if no positives found, just to have some signal.
            num_keep_neg = max(1, int(num_neg * 0.01))
            
        # Randomly sample negatives
        if num_keep_neg < num_neg and num_neg > 0:
            perm = torch.randperm(num_neg, device=neg_indices.device)
            keep_neg_indices = neg_indices[perm[:num_keep_neg]]
        else:
            keep_neg_indices = neg_indices
            
        # Combine
        combined_indices = torch.cat([pos_indices, keep_neg_indices])
        
        return combined_indices

    def class_ce_loss(self, y_pred, y_true):
        """Class-wise Cross Entropy loss"""
        y_true_indices = torch.argmax(y_true, dim=-1) # (B, T, G)
        
        B, T, G, M = y_pred.shape
        y_pred_flat = y_pred.view(-1, M) # (N, M)
        y_true_flat = y_true_indices.view(-1) # (N)
        
        if self.use_masked_loss:
            mask_indices = self.get_mask_indices(y_true)
            
            if mask_indices.numel() == 0:
                return torch.tensor(0.0, device=y_pred.device, require_grad=True)
                
            y_pred_masked = y_pred_flat[mask_indices]
            y_true_masked = y_true_flat[mask_indices]
            
            # Loss is already reduced='none' in init
            loss = self.ce_loss(y_pred_masked, y_true_masked)
            return loss.mean()
        else:
            # Fallback for full loss, but self.ce_loss is 'none' now, so we mean it manually
            loss = self.ce_loss(y_pred_flat, y_true_flat)
            return loss.mean()

    def class_mse_loss(self, y_pred, y_true):
        """Class-wise Mean Squared Error loss"""
        y_pred_probs = F.softmax(y_pred, dim=-1)
        
        if self.use_masked_loss:
            # Use same masking logic
            # MSE requires matching dimensions
            B, T, G, M = y_pred.shape
            y_pred_flat = y_pred_probs.view(-1, M)
            y_true_flat = y_true.view(-1, M)
            
            mask_indices = self.get_mask_indices(y_true)
            
            if mask_indices.numel() == 0:
                 return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            
            y_pred_masked = y_pred_flat[mask_indices]
            y_true_masked = y_true_flat[mask_indices]
            
            mse_loss = F.mse_loss(y_pred_masked, y_true_masked)
            return mse_loss
            
        else:
            # MSE Loss
            mse_loss = F.mse_loss(y_pred_probs, y_true)
            return mse_loss
    
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
        
        if self.loss_type == 'mse':
            loss_class = self.class_mse_loss(y_pred, y_true)
        else:
            loss_class = self.class_ce_loss(y_pred, y_true)
        
        # For other losses (AIUR, CL), we need probabilities
        # y_pred_probs = F.softmax(y_pred, dim=-1)
        
        # loss_aiur = self.aiur_loss(y_pred_probs, y_true)
        # loss_cl = self.converging_localization_loss(y_pred_probs, y_true)
        
        total_loss = self.w_class * loss_class
        
        # Add Truncated L1 Loss
        loss_trunc = torch.tensor(0.0, device=y_pred.device)
        if self.w_trunc_l1 > 0:
            loss_trunc = self.truncated_l1_loss(y_pred)
            total_loss += self.w_trunc_l1 * loss_trunc
            
        breakdown = {
            f'class_{self.loss_type}': float(loss_class.item()),
            # 'aiur': float(loss_aiur.item()),
            # 'cl': float(loss_cl.item())
        }
        
        if self.w_trunc_l1 > 0:
            breakdown['trunc_l1'] = float(loss_trunc.item())
            
        return total_loss, breakdown
