"""
CycleGAN model definition.
Implements generators and discriminators for unpaired image-to-image translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""
    
    def __init__(self, in_channels: int):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    """
    Generator network for CycleGAN.
    Uses ResNet-based architecture with skip connections.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        n_residual_blocks: int = 9
    ):
        """
        Initialize Generator.
        
        Args:
            input_channels: Number of input image channels
            output_channels: Number of output image channels
            n_residual_blocks: Number of residual blocks in the generator
        """
        super(Generator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_features = 64
        for _ in range(2):
            out_features = in_features * 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # Upsampling
        for _ in range(2):
            out_features = in_features // 2
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    """
    Discriminator network for CycleGAN.
    Uses PatchGAN architecture.
    """
    
    def __init__(self, input_channels: int = 1):
        """
        Initialize Discriminator.
        
        Args:
            input_channels: Number of input image channels
        """
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters: int, out_filters: int, normalize: bool = True):
            """Discriminator block."""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CycleGAN(nn.Module):
    """
    Complete CycleGAN model.
    Contains two generators and two discriminators.
    """
    
    def __init__(
        self,
        input_channels_a: int = 1,
        input_channels_b: int = 1,
        n_residual_blocks: int = 9
    ):
        """
        Initialize CycleGAN model.
        
        Args:
            input_channels_a: Number of channels for domain A images
            input_channels_b: Number of channels for domain B images
            n_residual_blocks: Number of residual blocks in generators
        """
        super(CycleGAN, self).__init__()
        
        # Generators
        self.G_A2B = Generator(input_channels_a, input_channels_b, n_residual_blocks)
        self.G_B2A = Generator(input_channels_b, input_channels_a, n_residual_blocks)
        
        # Discriminators
        self.D_A = Discriminator(input_channels_a)
        self.D_B = Discriminator(input_channels_b)
    
    def forward(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through CycleGAN.
        
        Args:
            real_a: Real images from domain A [batch_size, channels, H, W]
            real_b: Real images from domain B [batch_size, channels, H, W]
        
        Returns:
            Tuple containing:
                - fake_b: Generated images in domain B
                - fake_a: Generated images in domain A
                - rec_a: Reconstructed images in domain A (cycle consistency)
                - rec_b: Reconstructed images in domain B (cycle consistency)
                - idt_a: Identity mapping for domain A (optional)
                - idt_b: Identity mapping for domain B (optional)
        """
        # Generate fake images
        fake_b = self.G_A2B(real_a)
        fake_a = self.G_B2A(real_b)
        
        # Cycle consistency: reconstruct original images
        rec_a = self.G_B2A(fake_b)
        rec_b = self.G_A2B(fake_a)
        
        # Identity mapping (optional, for training stability)
        idt_a = self.G_B2A(real_a)
        idt_b = self.G_A2B(real_b)
        
        return fake_b, fake_a, rec_a, rec_b, idt_a, idt_b


def create_cyclegan_model(
    input_channels_a: int = 1,
    input_channels_b: int = 1,
    n_residual_blocks: int = 9
) -> CycleGAN:
    """
    Factory function to create a CycleGAN model.
    
    Args:
        input_channels_a: Number of channels for domain A
        input_channels_b: Number of channels for domain B
        n_residual_blocks: Number of residual blocks in generators
    
    Returns:
        Initialized CycleGAN model
    """
    return CycleGAN(
        input_channels_a=input_channels_a,
        input_channels_b=input_channels_b,
        n_residual_blocks=n_residual_blocks
    )

