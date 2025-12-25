"""
Training script for nodule segmentation using DinoV2 backbone.
Designed for use in Jupyter notebooks.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image

from dataloader import (
    create_ddti_dataloader,
    get_default_transform,
    calculate_mean_std_ddti
)
from segmentation_model import (
    create_segmentation_model,
    CombinedSegmentationLoss,
    train_segmentation_epoch,
    validate_segmentation,
    visualize_segmentation_results
)


def create_segmentation_transforms(image_size: int = 224, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
    """
    Create transforms for image and mask for segmentation.
    
    Args:
        image_size: Target image size
        mean: Mean values for normalization
        std: Std values for normalization
        
    Returns:
        Transform function that works on dict with 'image' and 'mask' keys
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    def transform_fn(sample):
        """Transform function for segmentation samples."""
        img = sample['image']
        mask = sample['mask']
        
        # Resize both image and mask
        img = img.resize((image_size, image_size), Image.BILINEAR)
        mask = mask.resize((image_size, image_size), Image.NEAREST)
        
        # Convert image to tensor and normalize
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        img_tensor = img_transform(img)
        
        # Convert mask to tensor (keep as single channel)
        mask_transform = transforms.ToTensor()
        mask_tensor = mask_transform(mask)
        
        # Binarize mask (threshold at 0.5)
        mask_tensor = (mask_tensor > 0.5).float()
        
        sample['image'] = img_tensor
        sample['mask'] = mask_tensor
        
        return sample
    
    return transform_fn


def train_segmentation_model(
    # Data arguments
    train_root: str,
    val_root: Optional[str] = None,
    image_size: int = 224,
    
    # Model arguments
    model_name: str = 'facebook/dinov2-small',
    num_classes: int = 1,
    freeze_backbone: bool = False,
    
    # Training arguments
    batch_size: int = 16,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    dice_weight: float = 0.5,
    bce_weight: float = 0.5,
    num_workers: int = 4,
    
    # Checkpoint arguments
    checkpoint_dir: str = './checkpoints',
    resume: Optional[str] = None,
    save_every: int = 5,
    
    # Visualization
    visualize_every: int = 5,
    visualize_dir: Optional[str] = None,
    
    # Device
    device: Optional[str] = None
):
    """
    Train DinoV2 segmentation model for nodule segmentation.
    
    This function can be called directly from Jupyter notebook cells.
    
    Args:
        train_root: Root directory for training data (contains image/ and masks/ subdirectories)
        val_root: Root directory for validation data (contains image/ and masks/ subdirectories)
        image_size: Image size for training (default: 224)
        
        model_name: DinoV2 model name (default: 'facebook/dinov2-small')
        num_classes: Number of output classes (default: 1 for binary segmentation)
        freeze_backbone: Freeze the backbone during training (default: False)
        
        batch_size: Batch size (default: 16)
        epochs: Number of epochs (default: 50)
        learning_rate: Learning rate (default: 1e-4)
        weight_decay: Weight decay (default: 1e-4)
        dice_weight: Weight for Dice loss (default: 0.5)
        bce_weight: Weight for BCE loss (default: 0.5)
        num_workers: Number of data loader workers (default: 4)
        
        checkpoint_dir: Directory to save checkpoints (default: './checkpoints')
        resume: Path to checkpoint to resume from (default: None)
        save_every: Save checkpoint every N epochs (default: 5)
        
        visualize_every: Visualize results every N epochs (default: 5)
        visualize_dir: Directory to save visualizations (default: None, uses checkpoint_dir/visualizations)
        
        device: Device to use ('auto', 'cpu', 'cuda', or None for auto) (default: None)
    
    Returns:
        Trained model
    """
    # Set device
    if device is None or device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create visualization directory
    if visualize_dir is None:
        visualize_dir_path = checkpoint_dir_path / 'visualizations'
    else:
        visualize_dir_path = Path(visualize_dir)
    visualize_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate mean and std from training dataset
    print("Calculating mean and std from training dataset...")
    mean, std = calculate_mean_std_ddti(
        root=train_root,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Create transforms
    train_transform = create_segmentation_transforms(
        image_size=image_size,
        mean=mean,
        std=std
    )
    val_transform = create_segmentation_transforms(
        image_size=image_size,
        mean=mean,
        std=std
    )
    
    # Create data loaders
    print("Creating training data loader...")
    train_loader = create_ddti_dataloader(
        root=train_root,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        transform=train_transform,
        return_size=False
    )
    
    val_loader = None
    if val_root:
        print("Creating validation data loader...")
        val_loader = create_ddti_dataloader(
            root=val_root,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            transform=val_transform,
            return_size=False
        )
    
    # Create model
    print(f"Creating segmentation model: {model_name}")
    model = create_segmentation_model(
        model_name=model_name,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone
    )
    model = model.to(device)
    
    # Create loss function
    criterion = CombinedSegmentationLoss(
        dice_weight=dice_weight,
        bce_weight=bce_weight
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume:
        print(f"Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("Starting training...")
    best_val_dice = 0.0
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_dice = train_segmentation_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        
        # Validate
        if val_loader is not None:
            val_loss, val_dice = validate_segmentation(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device
            )
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            
            # Save best model
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_checkpoint_path = checkpoint_dir_path / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_dice,
                    'val_loss': val_loss,
                }, best_checkpoint_path)
                print(f"Saved best model (val_dice: {val_dice:.4f})")
        
        # Visualize results periodically
        if (epoch + 1) % visualize_every == 0 and val_loader is not None:
            print(f"\nVisualizing segmentation results at epoch {epoch + 1}...")
            vis_path = visualize_dir_path / f'segmentation_epoch_{epoch + 1}.png'
            visualize_segmentation_results(
                model=model,
                dataloader=val_loader,
                device=device,
                num_samples=4,
                save_path=str(vis_path)
            )
        
        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            checkpoint_path = checkpoint_dir_path / f'checkpoint_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_dice': train_dice,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_checkpoint_path = checkpoint_dir_path / 'final_model.pt'
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_dice': train_dice,
    }, final_checkpoint_path)
    print(f"\nTraining completed! Final model saved to {final_checkpoint_path}")
    
    return model

