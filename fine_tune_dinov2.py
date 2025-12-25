"""
Fine-tuning module for DinoV2 model on ultrasound images.
Uses self-supervised learning with contrastive learning approach.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Dinov2Model, Dinov2Config
from typing import Optional, Dict, Any
import os
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F


class DinoV2FineTuner(nn.Module):
    """
    Fine-tuned DinoV2 model for ultrasound image analysis.
    Uses self-supervised learning with contrastive learning.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        projection_dim: int = 128,
        freeze_backbone: bool = False
    ):
        """
        Initialize the DinoV2 fine-tuner.
        
        Args:
            model_name: HuggingFace model name for DinoV2
            projection_dim: Dimension of the projection head output
            freeze_backbone: Whether to freeze the backbone during training
        """
        super().__init__()
        
        # Load DinoV2 model
        self.backbone = Dinov2Model.from_pretrained(model_name)
        self.embed_dim = self.backbone.config.hidden_size
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head for contrastive learning
        # This maps the DinoV2 features to a lower-dimensional space
        self.projection_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, projection_dim),
            L2Norm(dim=-1)  # Normalize for cosine similarity
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input image tensor [batch_size, 3, H, W] (RGB images)
            
        Returns:
            Projected features [batch_size, projection_dim]
        """
        # Ensure input is on the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)
        
        # Get features from DinoV2
        # DinoV2 expects pixel_values as keyword argument
        outputs = self.backbone(pixel_values=x)
        # Use CLS token (first token) for image-level representation
        features = outputs.last_hidden_state[:, 0]  # CLS token
        
        # Project to lower dimension
        projected = self.projection_head(features)
        
        return projected


class L2Norm(nn.Module):
    """L2 normalization layer."""
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=self.dim)


def create_fine_tuner(
    model_name: str = "facebook/dinov2-small",
    projection_dim: int = 128,
    freeze_backbone: bool = False
) -> DinoV2FineTuner:
    """
    Create a DinoV2 fine-tuner model.
    
    Args:
        model_name: HuggingFace model name for DinoV2
        projection_dim: Dimension of the projection head output
        freeze_backbone: Whether to freeze the backbone during training
        
    Returns:
        DinoV2FineTuner instance
    """
    return DinoV2FineTuner(
        model_name=model_name,
        projection_dim=projection_dim,
        freeze_backbone=freeze_backbone
    )


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for self-supervised learning.
    Uses InfoNCE (NT-Xent) loss.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between two augmented views.
        
        Args:
            features1: Features from first augmentation [batch_size, projection_dim]
            features2: Features from second augmentation [batch_size, projection_dim]
            
        Returns:
            Contrastive loss value
        """
        batch_size = features1.size(0)
        
        # Concatenate features
        features = torch.cat([features1, features2], dim=0)  # [2*batch_size, projection_dim]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
        labels = torch.arange(batch_size, device=features.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, device=features.device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


def train_epoch(
    model: DinoV2FineTuner,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: ContrastiveLoss,
    device: torch.device,
    augment_transform: Optional[torch.nn.Module] = None
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: DinoV2 fine-tuner model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        augment_transform: Optional augmentation transform
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Handle both dict batches (DDTI) and tensor batches
        if isinstance(batch, dict):
            images = batch['image'].to(device)
        else:
            images = batch.to(device)
        
        # Create two augmented views
        if augment_transform:
            images1 = augment_transform(images)
            images2 = augment_transform(images)
            # Ensure augmented images are on the correct device
            images1 = images1.to(device)
            images2 = images2.to(device)
        else:
            # Simple augmentation: use same images (for testing)
            images1 = images
            images2 = images
        
        # Forward pass
        features1 = model(images1)
        features2 = model(images2)
        
        # Compute loss
        loss = criterion(features1, features2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(
    model: DinoV2FineTuner,
    val_loader: DataLoader,
    criterion: ContrastiveLoss,
    device: torch.device,
    augment_transform: Optional[torch.nn.Module] = None
) -> float:
    """
    Validate the model.
    
    Args:
        model: DinoV2 fine-tuner model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        augment_transform: Optional augmentation transform
        
    Returns:
        Average loss for validation
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            # Handle both dict batches (DDTI) and tensor batches
            if isinstance(batch, dict):
                images = batch['image'].to(device)
            else:
                images = batch.to(device)
            
            # Create two augmented views
            if augment_transform:
                images1 = augment_transform(images)
                images2 = augment_transform(images)
                # Ensure augmented images are on the correct device
                images1 = images1.to(device)
                images2 = images2.to(device)
            else:
                images1 = images
                images2 = images
            
            # Forward pass
            features1 = model(images1)
            features2 = model(images2)
            
            # Compute loss
            loss = criterion(features1, features2)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(
    model: DinoV2FineTuner,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_path: str
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        checkpoint_path: Path to save checkpoint
    """
    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    model: DinoV2FineTuner,
    optimizer: Optional[optim.Optimizer],
    checkpoint_path: str
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

