"""
Simple classifier for FetalPlanesDB dataset.
Classifies images based on Brain_plane field.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import Counter
from PIL import Image
import torchvision.transforms as transforms

from ..fpd_dataset import FetalPlanesDBDataset


class SimpleBrainPlaneClassifier(nn.Module):
    """
    Simple CNN classifier for Brain_plane classification.
    """
    
    def __init__(self, num_classes: int, input_channels: int = 1):
        """
        Initialize classifier.
        
        Args:
            num_classes: Number of Brain_plane classes
            input_channels: Number of input image channels (1 for grayscale)
        """
        super(SimpleBrainPlaneClassifier, self).__init__()
        
        # Simple CNN architecture
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 112
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28 -> 14
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image tensor [batch_size, channels, H, W]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class BrainPlaneClassifierTrainer:
    """
    Trainer for Brain_plane classifier.
    """
    
    def __init__(
        self,
        model: SimpleBrainPlaneClassifier,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        """
        Initialize trainer.
        
        Args:
            model: Classifier model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training DataLoader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # Extract image and label
            images = batch['image'].to(self.device)
            labels = batch['label_idx'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100.0 * correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation DataLoader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                # Extract image and label
                images = batch['image'].to(self.device)
                labels = batch['label_idx'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100.0 * correct / total
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def save_checkpoint(self, checkpoint_path: str, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch']


def get_class_mapping(dataset: FetalPlanesDBDataset) -> Dict[str, int]:
    """
    Get mapping from Brain_plane class names to indices.
    Uses the dataset's labels directly for efficiency.
    
    Args:
        dataset: FetalPlanesDBDataset instance
        
    Returns:
        Dictionary mapping class names to indices
    """
    # Collect all unique Brain_plane values from dataset labels
    brain_planes = [label.get('Brain_plane', '') for label in dataset.labels]
    
    # Get unique classes and sort for consistency
    unique_classes = sorted(set(brain_planes))
    
    # Create mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    
    print(f"Found {len(unique_classes)} Brain_plane classes:")
    for cls, idx in class_to_idx.items():
        count = brain_planes.count(cls)
        print(f"  {cls}: {idx} ({count} samples)")
    
    return class_to_idx


def train_classifier(
    train_dataset: FetalPlanesDBDataset,
    val_dataset: Optional[FetalPlanesDBDataset] = None,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = './checkpoints/brain_plane_classifier',
    save_every: int = 10
):
    """
    Train Brain_plane classifier.
    
    Args:
        train_dataset: Training dataset (FetalPlanesDBDataset)
        val_dataset: Validation dataset (optional, if None uses train split)
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Device to train on (auto-detected if None)
        checkpoint_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Get class mapping
    print("Building class mapping...")
    class_to_idx = get_class_mapping(train_dataset)
    num_classes = len(class_to_idx)
    
    # Create classification datasets using FetalPlanesDBDataset with class_to_idx
    train_classifier_dataset = FetalPlanesDBDataset(
        root=str(train_dataset.root),
        class_to_idx=class_to_idx,
        transform=train_dataset.transform,
        target_size=train_dataset.target_size,
        csv_file=train_dataset.csv_path.name,
        images_dir=train_dataset.images_dir.name,
        train=True
    )
    
    if val_dataset is None:
        # Use validation split from original dataset
        val_classifier_dataset = FetalPlanesDBDataset(
            root=str(train_dataset.root),
            class_to_idx=class_to_idx,
            transform=train_dataset.transform,
            target_size=train_dataset.target_size,
            csv_file=train_dataset.csv_path.name,
            images_dir=train_dataset.images_dir.name,
            train=False
        )
    else:
        val_classifier_dataset = FetalPlanesDBDataset(
            root=str(val_dataset.root),
            class_to_idx=class_to_idx,
            transform=val_dataset.transform,
            target_size=val_dataset.target_size,
            csv_file=val_dataset.csv_path.name,
            images_dir=val_dataset.images_dir.name,
            train=False if hasattr(val_dataset, 'df') and 'Train' in val_dataset.df.columns else None
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_classifier_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_classifier_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print(f"Creating classifier with {num_classes} classes...")
    model = SimpleBrainPlaneClassifier(
        num_classes=num_classes,
        input_channels=1  # Grayscale images
    )
    
    # Create trainer
    trainer = BrainPlaneClassifierTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        trainer.history['train_loss'].append(train_metrics['loss'])
        trainer.history['train_acc'].append(train_metrics['accuracy'])
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        trainer.history['val_loss'].append(val_metrics['loss'])
        trainer.history['val_acc'].append(val_metrics['accuracy'])
        
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
        
        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
        
        if (epoch + 1) % save_every == 0 or is_best:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            trainer.save_checkpoint(str(checkpoint_path), epoch)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            if is_best:
                best_path = checkpoint_dir / 'best_model.pt'
                trainer.save_checkpoint(str(best_path), epoch)
                print(f"Best model saved to {best_path}")
    
    print("\nTraining completed!")
    return trainer, class_to_idx

