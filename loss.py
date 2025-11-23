import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SMRSELDLoss(nn.Module):
    """Complete SMR-SELD loss function with three components"""

    def __init__(self, loss_type='ce', w_class=1.0, w_aiur=0.5, w_cl=0.5, grid_size=None, class_weights=None):
        super().__init__()
        self.loss_type = loss_type
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

    def class_mse_loss(self, y_pred, y_true):
        """Class-wise Mean Squared Error loss
        y_pred: (B, T, G, M) - Logits
        y_true: (B, T, G, M) - One-hot encoded targets
        """
        # Apply Softmax to get probabilities
        y_pred_probs = F.softmax(y_pred, dim=-1)
        
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
        
        # total_loss = (self.w_class * loss_class +
        #               self.w_aiur * loss_aiur +
        #               self.w_cl * loss_cl)
        total_loss = self.w_class * loss_class  
        breakdown = {
            f'class_{self.loss_type}': float(loss_class.item()),
            # 'aiur': float(loss_aiur.item()),
            # 'cl': float(loss_cl.item())
        }
        return total_loss, breakdown
