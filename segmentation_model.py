"""
Segmentation model for nodule segmentation using DinoV2 as backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model
from typing import Optional


class DinoV2SegmentationModel(nn.Module):
    """
    Segmentation model using DinoV2 as backbone for nodule segmentation.
    Uses a decoder head to upsample features to original image resolution.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        num_classes: int = 1,
        freeze_backbone: bool = False
    ):
        """
        Initialize the DinoV2 segmentation model.
        
        Args:
            model_name: HuggingFace model name for DinoV2
            num_classes: Number of output classes (1 for binary segmentation)
            freeze_backbone: Whether to freeze the backbone during training
        """
        super().__init__()
        
        # Load DinoV2 model as backbone
        self.backbone = Dinov2Model.from_pretrained(model_name)
        self.embed_dim = self.backbone.config.hidden_size
        self.patch_size = self.backbone.config.patch_size
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Decoder head for segmentation
        # DinoV2 outputs features at patch level, we need to upsample to pixel level
        # The feature map size is (H/patch_size, W/patch_size)
        
        # Upsampling layers
        self.decoder = nn.Sequential(
            # First upsampling: 2x
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.embed_dim // 2),
            nn.ReLU(inplace=True),
            
            # Second upsampling: 2x
            nn.ConvTranspose2d(self.embed_dim // 2, self.embed_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.embed_dim // 4),
            nn.ReLU(inplace=True),
            
            # Third upsampling: 2x (if patch_size is 14, we need 3 upsamplings to get close to original)
            nn.ConvTranspose2d(self.embed_dim // 4, self.embed_dim // 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.embed_dim // 8),
            nn.ReLU(inplace=True),
            
            # Final classification layer
            nn.Conv2d(self.embed_dim // 8, num_classes, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input image tensor [batch_size, 3, H, W] (RGB images)
            
        Returns:
            Segmentation logits [batch_size, num_classes, H', W']
            Note: H' and W' may differ from H and W due to patch-based processing
        """
        # Ensure input is on the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)
        
        # Get features from DinoV2
        # DinoV2 expects pixel_values as keyword argument
        outputs = self.backbone(pixel_values=x)
        
        # Get patch embeddings (excluding CLS token)
        # DinoV2 outputs: [batch_size, num_patches + 1, embed_dim]
        # We need to reshape to spatial format: [batch_size, embed_dim, H_patch, W_patch]
        patch_embeddings = outputs.last_hidden_state[:, 1:]  # Remove CLS token [batch_size, num_patches, embed_dim]
        
        # Get spatial dimensions
        batch_size = patch_embeddings.size(0)
        num_patches = patch_embeddings.size(1)
        embed_dim = patch_embeddings.size(2)
        
        # Calculate spatial dimensions (assuming square images for simplicity)
        # DinoV2-small uses patch_size=14, so for 224x224 image: 224/14 = 16 patches per side
        H_patch = W_patch = int(num_patches ** 0.5)
        
        # Reshape to spatial format
        patch_embeddings = patch_embeddings.view(batch_size, H_patch, W_patch, embed_dim)
        patch_embeddings = patch_embeddings.permute(0, 3, 1, 2)  # [batch_size, embed_dim, H_patch, W_patch]
        
        # Decode to segmentation map
        seg_logits = self.decoder(patch_embeddings)
        
        # Upsample to match input size if needed
        # DinoV2-small: 224x224 -> 16x16 patches -> after 3x2 upsampling: 128x128
        # We need to upsample to original size
        _, _, H_out, W_out = seg_logits.shape
        _, _, H_in, W_in = x.shape
        
        if H_out != H_in or W_out != W_in:
            seg_logits = F.interpolate(
                seg_logits, 
                size=(H_in, W_in), 
                mode='bilinear', 
                align_corners=False
            )
        
        return seg_logits


def create_segmentation_model(
    model_name: str = "facebook/dinov2-small",
    num_classes: int = 1,
    freeze_backbone: bool = False
) -> DinoV2SegmentationModel:
    """
    Create a DinoV2 segmentation model.
    
    Args:
        model_name: HuggingFace model name for DinoV2
        num_classes: Number of output classes (1 for binary segmentation)
        freeze_backbone: Whether to freeze the backbone during training
        
    Returns:
        DinoV2SegmentationModel instance
    """
    return DinoV2SegmentationModel(
        model_name=model_name,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone
    )


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    """
    
    def __init__(self, smooth: float = 1e-6):
        """
        Initialize Dice loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predicted logits [batch_size, 1, H, W]
            target: Target mask [batch_size, 1, H, W] (values 0 or 1)
            
        Returns:
            Dice loss value
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1 - dice


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for segmentation: Dice loss + Binary Cross Entropy.
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-6):
        """
        Initialize combined segmentation loss.
        
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            smooth: Smoothing factor for Dice loss
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted logits [batch_size, 1, H, W]
            target: Target mask [batch_size, 1, H, W] (values 0 or 1)
            
        Returns:
            Combined loss value
        """
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        return self.dice_weight * dice + self.bce_weight * bce


def train_segmentation_epoch(
    model: DinoV2SegmentationModel,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedSegmentationLoss,
    device: torch.device
) -> tuple:
    """
    Train segmentation model for one epoch.
    
    Args:
        model: Segmentation model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        
    Returns:
        Tuple of (average_loss, average_dice_score)
    """
    from tqdm import tqdm
    
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Handle dict batches (DDTI)
        if isinstance(batch, dict):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
        else:
            raise ValueError("Expected dict batch with 'image' and 'mask' keys")
        
        # Forward pass
        pred_logits = model(images)
        
        # Compute loss
        loss = criterion(pred_logits, masks)
        
        # Calculate Dice score for monitoring
        with torch.no_grad():
            pred_probs = torch.sigmoid(pred_logits)
            dice_loss_fn = DiceLoss()
            dice_score = 1 - dice_loss_fn(pred_logits, masks)
            total_dice += dice_score.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': loss.item(),
            'dice': dice_score.item()
        })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_dice = total_dice / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_dice


def validate_segmentation(
    model: DinoV2SegmentationModel,
    val_loader,
    criterion: CombinedSegmentationLoss,
    device: torch.device
) -> tuple:
    """
    Validate segmentation model.
    
    Args:
        model: Segmentation model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average_loss, average_dice_score)
    """
    from tqdm import tqdm
    
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            # Handle dict batches (DDTI)
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
            else:
                raise ValueError("Expected dict batch with 'image' and 'mask' keys")
            
            # Forward pass
            pred_logits = model(images)
            
            # Compute loss
            loss = criterion(pred_logits, masks)
            
            # Calculate Dice score
            dice_loss_fn = DiceLoss()
            dice_score = 1 - dice_loss_fn(pred_logits, masks)
            
            total_loss += loss.item()
            total_dice += dice_score.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': loss.item(),
                'dice': dice_score.item()
            })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_dice = total_dice / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_dice


def visualize_segmentation_results(
    model: DinoV2SegmentationModel,
    dataloader,
    device: torch.device,
    num_samples: int = 4,
    save_path: Optional[str] = None
):
    """
    Visualize segmentation results.
    
    Args:
        model: Segmentation model
        dataloader: Data loader
        device: Device
        num_samples: Number of samples to visualize
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    model.eval()
    
    # Get a batch
    batch = next(iter(dataloader))
    
    if isinstance(batch, dict):
        images = batch['image'][:num_samples].to(device)
        masks = batch['mask'][:num_samples].to(device)
        image_names = batch.get('image_name', [f'Image {i}' for i in range(num_samples)])[:num_samples]
    else:
        raise ValueError("Expected dict batch")
    
    with torch.no_grad():
        pred_logits = model(images)
        pred_probs = torch.sigmoid(pred_logits)
        pred_masks = (pred_probs > 0.5).float()
    
    # Move to CPU for visualization
    images = images.cpu()
    masks = masks.cpu()
    pred_masks = pred_masks.cpu()
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original: {image_names[i]}')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        mask = masks[i][0].numpy()
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Predicted mask
        pred_mask = pred_masks[i][0].numpy()
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Segmentation visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


