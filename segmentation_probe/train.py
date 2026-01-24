"""
Main training script for fine-tuning DinoV2 on ultrasound images.

Designed for use in Jupyter notebooks. Import and call train_model() directly from notebook cells.

Example:
    from train import train_model
    
    train_model(
        train_root='./data/train',
        val_root='./data/val',
        use_ddti=True,
        batch_size=32,
        epochs=50,
        learning_rate=1e-4
    )
"""

import torch
import torch.nn as nn
from torchvision import transforms
import argparse
from pathlib import Path
from typing import List, Optional

from dataloader import (
    create_dataloader, 
    create_ddti_dataloader,
    get_default_transform, 
    calculate_mean_std,
    calculate_mean_std_ddti
)
from fine_tune_dinov2 import (
    create_fine_tuner,
    ContrastiveLoss,
    train_epoch,
    validate,
    save_checkpoint,
    load_checkpoint
)


class AugmentationTransform(nn.Module):
    """
    Augmentation transform for self-supervised learning.
    Applies random augmentations to create different views of the same image.
    Works with batches of tensors.
    """
    
    def __init__(self, image_size: int = 224, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
        super().__init__()
        self.image_size = image_size
        self.to_pil = transforms.ToPILImage(mode='RGB')  # Mode 'RGB' for color images
        
        # Use provided mean/std or ImageNet defaults
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        self.mean = mean
        self.std = std
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to a batch of RGB images.
        
        Args:
            x: Batch of images [batch_size, 3, H, W] (RGB images)
            
        Returns:
            Augmented batch [batch_size, 3, image_size, image_size]
        """
        batch_size = x.size(0)
        augmented = []
        
        # Create augmentation transform for RGB images
        transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            # For RGB: brightness, contrast, saturation, and hue
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Apply augmentation to each image in the batch
        for i in range(batch_size):
            img = x[i]
            assert img.size(0) == 3, "Image must be 3-channel RGB"
            # Convert RGB tensor to PIL
            img_pil = self.to_pil(img)
            img_aug = transform(img_pil)
            augmented.append(img_aug)
        
        return torch.stack(augmented, dim=0)


def create_augmentation_transform(image_size: int = 224, mean: Optional[List[float]] = None, std: Optional[List[float]] = None) -> AugmentationTransform:
    """Create augmentation transform for training."""
    return AugmentationTransform(image_size=image_size, mean=mean, std=std)


def train_model(
    # Data arguments
    train_folders: Optional[List[str]] = None,
    train_root: Optional[str] = None,
    val_folders: Optional[List[str]] = None,
    val_root: Optional[str] = None,
    use_ddti: bool = False,
    image_size: int = 224,
    
    # Model arguments
    model_name: str = 'facebook/dinov2-small',
    projection_dim: int = 128,
    freeze_backbone: bool = False,
    
    # Training arguments
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    num_workers: int = 4,
    
    # Checkpoint arguments
    checkpoint_dir: str = './checkpoints',
    resume: Optional[str] = None,
    save_every: int = 5,
    
    # Device
    device: Optional[str] = None
):
    """
    Train DinoV2 model on ultrasound images.
    
    This function can be called directly from Jupyter notebook cells.
    
    Args:
        train_folders: List of folders containing training images (legacy mode)
        train_root: Root directory for DDTI-style training data (contains image/ subdirectory)
        val_folders: List of folders containing validation images (legacy mode)
        val_root: Root directory for DDTI-style validation data (contains image/ subdirectory)
        use_ddti: Use DDTI-style directory structure (root/image/)
        image_size: Image size for training (default: 224)
        
        model_name: DinoV2 model name (default: 'facebook/dinov2-small')
        projection_dim: Projection head dimension (default: 128)
        freeze_backbone: Freeze the backbone during training (default: False)
        
        batch_size: Batch size (default: 32)
        epochs: Number of epochs (default: 50)
        learning_rate: Learning rate (default: 1e-4)
        weight_decay: Weight decay (default: 1e-4)
        temperature: Temperature for contrastive loss (default: 0.07)
        num_workers: Number of data loader workers (default: 4)
        
        checkpoint_dir: Directory to save checkpoints (default: './checkpoints')
        resume: Path to checkpoint to resume from (default: None)
        save_every: Save checkpoint every N epochs (default: 5)
        
        device: Device to use ('auto', 'cpu', 'cuda', or None for auto) (default: None)
    
    Returns:
        Trained model and training history (if needed)
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
    
    # Create model
    print(f"Creating model: {model_name}")
    model = create_fine_tuner(
        model_name=model_name,
        projection_dim=projection_dim,
        freeze_backbone=freeze_backbone
    )
    model = model.to(device)
    
    # Create loss function
    criterion = ContrastiveLoss(temperature=temperature)
    
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
        checkpoint = load_checkpoint(model, optimizer, resume)
        start_epoch = checkpoint['epoch'] + 1
    
    # Determine if using DDTI structure
    use_ddti_mode = use_ddti or train_root is not None
    
    if use_ddti_mode:
        if train_root is None:
            raise ValueError("train_root must be specified when using DDTI structure")
        
        # Calculate mean and std from training dataset (DDTI)
        print("Calculating mean and std from DDTI training dataset...")
        mean, std = calculate_mean_std_ddti(
            root=train_root,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        # Create transforms for DDTI (works on dict samples)
        def create_ddti_train_transform():
            """Create transform for DDTI training (works on dict with 'image' key)."""
            def transform_fn(sample):
                """Transform function for DDTI sample dict."""
                img = sample['image']
                transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor()  # Converts RGB PIL to [3, H, W] tensor
                ])
                sample['image'] = transform(img)
                return sample
            
            return transform_fn
        
        def create_ddti_val_transform():
            """Create transform for DDTI validation (works on dict with 'image' key)."""
            val_transform_base = get_default_transform(
                image_size=image_size,
                mean=mean,
                std=std
            )
            
            def transform_fn(sample):
                """Transform function for DDTI sample dict."""
                img = sample['image']
                # Apply the transform (expects PIL Image)
                sample['image'] = val_transform_base(img)
                return sample
            
            return transform_fn
        
        train_transform = create_ddti_train_transform()
        val_transform_fn = create_ddti_val_transform()
        train_aug_transform = create_augmentation_transform(
            image_size=image_size,
            mean=mean,
            std=std
        )
        
        # Create DDTI data loaders
        print("Creating DDTI training data loader...")
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
            print("Creating DDTI validation data loader...")
            val_loader = create_ddti_dataloader(
                root=val_root,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                transform=val_transform_fn,
                return_size=False
            )
    else:
        # Legacy mode: use folder-based loading
        if train_folders is None:
            raise ValueError("Either train_folders or train_root must be specified")
        
        # Calculate mean and std from training dataset
        print("Calculating mean and std from training dataset...")
        mean, std = calculate_mean_std(
            image_folders=train_folders,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        # Create data transforms
        # For training, we need to load images as tensors first, then apply augmentation
        train_base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()  # Converts RGB PIL to [3, H, W] tensor
        ])
        train_aug_transform = create_augmentation_transform(
            image_size=image_size,
            mean=mean,
            std=std
        )
        val_transform = get_default_transform(
            image_size=image_size,
            mean=mean,
            std=std
        )
        
        # Create data loaders
        print("Creating training data loader...")
        train_loader = create_dataloader(
            image_folders=train_folders,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            transform=train_base_transform,  # Basic transform to tensor
            image_extensions=None
        )
        
        val_loader = None
        if val_folders:
            print("Creating validation data loader...")
            val_loader = create_dataloader(
                image_folders=val_folders,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                transform=val_transform,
                image_extensions=None
            )
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            augment_transform=train_aug_transform
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if val_loader is not None:
            val_loss = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                augment_transform=train_aug_transform
            )
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = checkpoint_dir_path / 'best_model.pt'
                save_checkpoint(model, optimizer, epoch, val_loss, str(best_checkpoint_path))
                print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            checkpoint_path = checkpoint_dir_path / f'checkpoint_epoch_{epoch + 1}.pt'
            save_checkpoint(model, optimizer, epoch, train_loss, str(checkpoint_path))
    
    # Save final model
    final_checkpoint_path = checkpoint_dir_path / 'final_model.pt'
    save_checkpoint(model, optimizer, epochs - 1, train_loss, str(final_checkpoint_path))
    print(f"\nTraining completed! Final model saved to {final_checkpoint_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Fine-tune DinoV2 on ultrasound images')
    
    # Data arguments
    parser.add_argument('--train_folders', type=str, nargs='+', default=None,
                        help='List of folders containing training images (legacy mode)')
    parser.add_argument('--train_root', type=str, default=None,
                        help='Root directory for DDTI-style training data (contains image/ subdirectory)')
    parser.add_argument('--val_folders', type=str, nargs='+', default=None,
                        help='List of folders containing validation images (legacy mode)')
    parser.add_argument('--val_root', type=str, default=None,
                        help='Root directory for DDTI-style validation data (contains image/ subdirectory)')
    parser.add_argument('--use_ddti', action='store_true',
                        help='Use DDTI-style directory structure (root/image/)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for training (default: 224)')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='facebook/dinov2-small',
                        help='DinoV2 model name (default: facebook/dinov2-small)')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Projection head dimension (default: 128)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze the backbone during training')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss (default: 0.07)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda) (default: auto)')
    
    args = parser.parse_args()
    
    # Call train_model with parsed arguments
    train_model(
        train_folders=args.train_folders,
        train_root=args.train_root,
        val_folders=args.val_folders,
        val_root=args.val_root,
        use_ddti=args.use_ddti or args.train_root is not None,
        image_size=args.image_size,
        model_name=args.model_name,
        projection_dim=args.projection_dim,
        freeze_backbone=args.freeze_backbone,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        save_every=args.save_every,
        device=args.device
    )


if __name__ == '__main__':
    main()

