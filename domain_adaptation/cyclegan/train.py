"""
Training script for CycleGAN.
Handles the training loop, loss computation, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import random

from .model import CycleGAN


class CycleGANTrainer:
    """
    Trainer class for CycleGAN.
    Handles training loop, loss computation, and model checkpointing.
    """
    
    def __init__(
        self,
        model: CycleGAN,
        device: torch.device,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999
    ):
        """
        Initialize CycleGAN trainer.
        
        Args:
            model: CycleGAN model instance
            device: Device to train on
            lambda_cycle: Weight for cycle consistency loss
            lambda_identity: Weight for identity loss
            lr_g: Learning rate for generators
            lr_d: Learning rate for discriminators
            beta1: Beta1 for Adam optimizer
            beta2: Beta2 for Adam optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        
        # Loss functions
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            list(model.G_A2B.parameters()) + list(model.G_B2A.parameters()),
            lr=lr_g,
            betas=(beta1, beta2)
        )
        self.optimizer_D_A = optim.Adam(
            model.D_A.parameters(),
            lr=lr_d,
            betas=(beta1, beta2)
        )
        self.optimizer_D_B = optim.Adam(
            model.D_B.parameters(),
            lr=lr_d,
            betas=(beta1, beta2)
        )
        
        # Training history
        self.history = {
            'loss_G': [],
            'loss_D_A': [],
            'loss_D_B': [],
            'loss_cycle': [],
            'loss_identity': []
        }
    
    def visualize_samples(
        self,
        dataloader: DataLoader,
        epoch: int,
        output_dir: str,
        num_samples: int = 1
    ):
        """
        Visualize generated samples from both domains.
        
        Args:
            dataloader: DataLoader for getting sample images
            epoch: Current epoch number
            output_dir: Directory to save visualization images
            num_samples: Number of sample pairs to visualize
        """
        self.model.eval()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get a random batch
        with torch.no_grad():
            # Get a random batch from dataloader
            all_batches = list(dataloader)
            if len(all_batches) == 0:
                return
            
            batch_idx = random.randint(0, len(all_batches) - 1)
            real_a, real_b = all_batches[batch_idx]
            real_a = real_a.to(self.device)
            real_b = real_b.to(self.device)
            
            # Take first image from batch (or up to num_samples)
            num_samples = min(num_samples, real_a.size(0), real_b.size(0))
            
            # Generate images
            fake_b, fake_a, rec_a, rec_b, _, _ = self.model(real_a[:num_samples], real_b[:num_samples])
            
            # Denormalize images from [-1, 1] to [0, 1] for visualization
            def denormalize(tensor):
                """Convert from [-1, 1] to [0, 1] range."""
                return (tensor + 1) / 2.0
            
            # Create visualization
            fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(num_samples):
                # Domain A -> B translation
                img_real_a = denormalize(real_a[i].cpu())
                img_fake_b = denormalize(fake_b[i].cpu())
                
                # Domain B -> A translation
                img_real_b = denormalize(real_b[i].cpu())
                img_fake_a = denormalize(fake_a[i].cpu())
                
                # Handle grayscale images (single channel)
                if img_real_a.dim() == 3 and img_real_a.size(0) == 1:
                    img_real_a = img_real_a.squeeze(0)
                    img_fake_b = img_fake_b.squeeze(0)
                    img_real_b = img_real_b.squeeze(0)
                    img_fake_a = img_fake_a.squeeze(0)
                    
                    # Convert to numpy
                    img_real_a = img_real_a.numpy()
                    img_fake_b = img_fake_b.numpy()
                    img_real_b = img_real_b.numpy()
                    img_fake_a = img_fake_a.numpy()
                    
                    # Plot Domain A -> B
                    axes[i, 0].imshow(img_real_a, cmap='gray', vmin=0, vmax=1)
                    axes[i, 0].set_title('Real A')
                    axes[i, 0].axis('off')
                    
                    axes[i, 1].imshow(img_fake_b, cmap='gray', vmin=0, vmax=1)
                    axes[i, 1].set_title('Fake B (A→B)')
                    axes[i, 1].axis('off')
                    
                    # Plot Domain B -> A
                    axes[i, 2].imshow(img_real_b, cmap='gray', vmin=0, vmax=1)
                    axes[i, 2].set_title('Real B')
                    axes[i, 2].axis('off')
                    
                    axes[i, 3].imshow(img_fake_a, cmap='gray', vmin=0, vmax=1)
                    axes[i, 3].set_title('Fake A (B→A)')
                    axes[i, 3].axis('off')
                else:
                    # Handle RGB images
                    if img_real_a.dim() == 3:
                        img_real_a = img_real_a.permute(1, 2, 0).numpy()
                        img_fake_b = img_fake_b.permute(1, 2, 0).numpy()
                        img_real_b = img_real_b.permute(1, 2, 0).numpy()
                        img_fake_a = img_fake_a.permute(1, 2, 0).numpy()
                    
                    # Plot Domain A -> B
                    axes[i, 0].imshow(np.clip(img_real_a, 0, 1))
                    axes[i, 0].set_title('Real A')
                    axes[i, 0].axis('off')
                    
                    axes[i, 1].imshow(np.clip(img_fake_b, 0, 1))
                    axes[i, 1].set_title('Fake B (A→B)')
                    axes[i, 1].axis('off')
                    
                    # Plot Domain B -> A
                    axes[i, 2].imshow(np.clip(img_real_b, 0, 1))
                    axes[i, 2].set_title('Real B')
                    axes[i, 2].axis('off')
                    
                    axes[i, 3].imshow(np.clip(img_fake_a, 0, 1))
                    axes[i, 3].set_title('Fake A (B→A)')
                    axes[i, 3].axis('off')
            
            plt.suptitle(f'Epoch {epoch+1} - Generated Samples', fontsize=16)
            plt.tight_layout()
            
            # Save figure
            save_path = output_dir / f'samples_epoch_{epoch+1}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Sample images saved to {save_path}")
        
        self.model.train()
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        n_epochs: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            epoch: Current epoch number
            n_epochs: Total number of epochs
            
        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()
        
        total_loss_G = 0.0
        total_loss_D_A = 0.0
        total_loss_D_B = 0.0
        total_loss_cycle = 0.0
        total_loss_identity = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for batch_idx, (real_a, real_b) in enumerate(pbar):
            real_a = real_a.to(self.device)
            real_b = real_b.to(self.device)
            
            # Adversarial ground truths
            valid = torch.ones(real_a.size(0), 1, 1, 1, device=self.device, requires_grad=False)
            fake = torch.zeros(real_a.size(0), 1, 1, 1, device=self.device, requires_grad=False)
            
            # ===================
            # Train Generators
            # ===================
            self.optimizer_G.zero_grad()
            
            # Generate images
            fake_b, fake_a, rec_a, rec_b, idt_a, idt_b = self.model(real_a, real_b)
            
            # GAN loss
            loss_G_A2B = self.criterion_gan(self.model.D_B(fake_b), valid)
            loss_G_B2A = self.criterion_gan(self.model.D_A(fake_a), valid)
            loss_GAN = (loss_G_A2B + loss_G_B2A) / 2
            
            # Cycle consistency loss
            loss_cycle_A = self.criterion_cycle(rec_a, real_a)
            loss_cycle_B = self.criterion_cycle(rec_b, real_b)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            
            # Identity loss
            loss_idt_A = self.criterion_identity(idt_a, real_a)
            loss_idt_B = self.criterion_identity(idt_b, real_b)
            loss_identity = (loss_idt_A + loss_idt_B) / 2
            
            # Total generator loss
            loss_G = loss_GAN + self.lambda_cycle * loss_cycle + self.lambda_identity * loss_identity
            
            loss_G.backward()
            self.optimizer_G.step()
            
            # ===================
            # Train Discriminator A
            # ===================
            self.optimizer_D_A.zero_grad()
            
            # Real loss
            loss_real = self.criterion_gan(self.model.D_A(real_a), valid)
            # Fake loss
            loss_fake = self.criterion_gan(self.model.D_A(fake_a.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2
            
            loss_D_A.backward()
            self.optimizer_D_A.step()
            
            # ===================
            # Train Discriminator B
            # ===================
            self.optimizer_D_B.zero_grad()
            
            # Real loss
            loss_real = self.criterion_gan(self.model.D_B(real_b), valid)
            # Fake loss
            loss_fake = self.criterion_gan(self.model.D_B(fake_b.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2
            
            loss_D_B.backward()
            self.optimizer_D_B.step()
            
            # Update totals
            total_loss_G += loss_G.item()
            total_loss_D_A += loss_D_A.item()
            total_loss_D_B += loss_D_B.item()
            total_loss_cycle += loss_cycle.item()
            total_loss_identity += loss_identity.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G': f'{loss_G.item():.4f}',
                'D_A': f'{loss_D_A.item():.4f}',
                'D_B': f'{loss_D_B.item():.4f}',
                'Cycle': f'{loss_cycle.item():.4f}',
                'Idt': f'{loss_identity.item():.4f}'
            })
        
        # Calculate averages
        n_batches = len(dataloader)
        avg_losses = {
            'loss_G': total_loss_G / n_batches,
            'loss_D_A': total_loss_D_A / n_batches,
            'loss_D_B': total_loss_D_B / n_batches,
            'loss_cycle': total_loss_cycle / n_batches,
            'loss_identity': total_loss_identity / n_batches
        }
        
        # Update history
        for key, value in avg_losses.items():
            self.history[key].append(value)
        
        return avg_losses
    
    def save_checkpoint(
        self,
        checkpoint_dir: str,
        epoch: int,
        is_best: bool = False
    ):
        """
        Save model checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_A_state_dict': self.optimizer_D_A.state_dict(),
            'optimizer_D_B_state_dict': self.optimizer_D_B.state_dict(),
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A_state_dict'])
        self.optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        return checkpoint['epoch']


def train_cyclegan(
    model: CycleGAN,
    train_loader: DataLoader,
    n_epochs: int = 200,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = './checkpoints/cyclegan',
    save_every: int = 10,
    lambda_cycle: float = 10.0,
    lambda_identity: float = 0.5,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    visualize_dir: Optional[str] = None
):
    """
    Main training function for CycleGAN.
    
    Args:
        model: CycleGAN model instance
        train_loader: Training DataLoader
        n_epochs: Number of training epochs
        device: Device to train on (auto-detected if None)
        checkpoint_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
        lambda_cycle: Weight for cycle consistency loss
        lambda_identity: Weight for identity loss
        lr_g: Learning rate for generators
        lr_d: Learning rate for discriminators
        visualize_dir: Directory to save visualization images (default: checkpoint_dir/samples)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Set up visualization directory
    if visualize_dir is None:
        visualize_dir = str(Path(checkpoint_dir) / 'samples')
    visualize_dir = Path(visualize_dir)
    visualize_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = CycleGANTrainer(
        model=model,
        device=device,
        lambda_cycle=lambda_cycle,
        lambda_identity=lambda_identity,
        lr_g=lr_g,
        lr_d=lr_d
    )
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(n_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"{'='*50}")
        
        # Train epoch
        losses = trainer.train_epoch(train_loader, epoch, n_epochs)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Generator Loss: {losses['loss_G']:.4f}")
        print(f"  Discriminator A Loss: {losses['loss_D_A']:.4f}")
        print(f"  Discriminator B Loss: {losses['loss_D_B']:.4f}")
        print(f"  Cycle Loss: {losses['loss_cycle']:.4f}")
        print(f"  Identity Loss: {losses['loss_identity']:.4f}")
        
        # Visualize generated samples
        print("Generating sample images...")
        trainer.visualize_samples(
            dataloader=train_loader,
            epoch=epoch,
            output_dir=str(visualize_dir),
            num_samples=1
        )
        
        # Save checkpoint
        is_best = losses['loss_G'] < best_loss
        if is_best:
            best_loss = losses['loss_G']
        
        if (epoch + 1) % save_every == 0 or is_best:
            trainer.save_checkpoint(checkpoint_dir, epoch, is_best)
            print(f"Checkpoint saved to {checkpoint_dir}")
    
    print("\nTraining completed!")
    return trainer

