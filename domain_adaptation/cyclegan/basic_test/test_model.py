"""
Simple test to verify CycleGAN model can be created and run forward pass.
"""

import unittest
import torch
from pathlib import Path
import sys

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import CycleGAN, Generator, Discriminator, create_cyclegan_model


class TestCycleGANModel(unittest.TestCase):
    """Test CycleGAN model creation and forward pass."""
    
    def test_generator_creation(self):
        """Test that Generator can be created."""
        generator = Generator(
            input_channels=1,
            output_channels=1,
            n_residual_blocks=9
        )
        
        self.assertIsNotNone(generator)
        self.assertTrue(isinstance(generator, Generator))
    
    def test_generator_forward_pass(self):
        """Test Generator forward pass."""
        generator = Generator(
            input_channels=1,
            output_channels=1,
            n_residual_blocks=9
        )
        
        # Create dummy input [batch, channels, height, width]
        batch_size = 2
        dummy_input = torch.randn(batch_size, 1, 256, 256)
        
        # Forward pass
        output = generator(dummy_input)
        
        # Check output shape
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], 1)  # Output channels
        self.assertEqual(output.shape[2], 256)  # Height
        self.assertEqual(output.shape[3], 256)  # Width
        self.assertEqual(output.shape, dummy_input.shape)
    
    def test_discriminator_creation(self):
        """Test that Discriminator can be created."""
        discriminator = Discriminator(input_channels=1)
        
        self.assertIsNotNone(discriminator)
        self.assertTrue(isinstance(discriminator, Discriminator))
    
    def test_discriminator_forward_pass(self):
        """Test Discriminator forward pass."""
        discriminator = Discriminator(input_channels=1)
        
        # Create dummy input [batch, channels, height, width]
        batch_size = 2
        dummy_input = torch.randn(batch_size, 1, 256, 256)
        
        # Forward pass
        output = discriminator(dummy_input)
        
        # Check output shape (PatchGAN outputs smaller feature map)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], 1)  # Single output channel
        # Height and width will be smaller due to downsampling
        self.assertGreater(output.shape[2], 0)
        self.assertGreater(output.shape[3], 0)
    
    def test_cyclegan_creation(self):
        """Test that CycleGAN model can be created."""
        model = CycleGAN(
            input_channels_a=1,
            input_channels_b=1,
            n_residual_blocks=9
        )
        
        self.assertIsNotNone(model)
        self.assertTrue(isinstance(model, CycleGAN))
        self.assertIsNotNone(model.G_A2B)
        self.assertIsNotNone(model.G_B2A)
        self.assertIsNotNone(model.D_A)
        self.assertIsNotNone(model.D_B)
    
    def test_cyclegan_forward_pass(self):
        """Test CycleGAN forward pass."""
        model = CycleGAN(
            input_channels_a=1,
            input_channels_b=1,
            n_residual_blocks=9
        )
        
        # Create dummy inputs
        batch_size = 2
        real_a = torch.randn(batch_size, 1, 256, 256)
        real_b = torch.randn(batch_size, 1, 256, 256)
        
        # Forward pass
        fake_b, fake_a, rec_a, rec_b, idt_a, idt_b = model(real_a, real_b)
        
        # Check output shapes
        self.assertEqual(fake_b.shape, real_b.shape)
        self.assertEqual(fake_a.shape, real_a.shape)
        self.assertEqual(rec_a.shape, real_a.shape)
        self.assertEqual(rec_b.shape, real_b.shape)
        self.assertEqual(idt_a.shape, real_a.shape)
        self.assertEqual(idt_b.shape, real_b.shape)
    
    def test_create_cyclegan_model_factory(self):
        """Test factory function for creating CycleGAN model."""
        model = create_cyclegan_model(
            input_channels_a=1,
            input_channels_b=1,
            n_residual_blocks=9
        )
        
        self.assertIsNotNone(model)
        self.assertTrue(isinstance(model, CycleGAN))
    
    def test_model_on_gpu_if_available(self):
        """Test that model can be moved to GPU if available."""
        model = create_cyclegan_model(
            input_channels_a=1,
            input_channels_b=1,
            n_residual_blocks=9
        )
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = model.to(device)
            
            # Create dummy inputs on GPU
            batch_size = 1
            real_a = torch.randn(batch_size, 1, 256, 256).to(device)
            real_b = torch.randn(batch_size, 1, 256, 256).to(device)
            
            # Forward pass on GPU
            fake_b, fake_a, rec_a, rec_b, idt_a, idt_b = model(real_a, real_b)
            
            # Check that outputs are on GPU
            self.assertTrue(fake_b.is_cuda)
            self.assertTrue(fake_a.is_cuda)
        else:
            print("CUDA not available, skipping GPU test")


if __name__ == '__main__':
    unittest.main()

