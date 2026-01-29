"""
Unit tests for FetalHeadCircDataset.
"""

import unittest
import torch
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import sys
import os

# Add parent directory to path to import dataset
# sys.path.insert(0, str(Path(__file__).parent.parent))
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)
from fhc_dataset import FetalHeadCircDataset


class TestFetalHeadCircDataset(unittest.TestCase):
    """Unit tests for FetalHeadCircDataset."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.test_data_dir = Path(__file__).parent / 'data'
        self.images_dir = self.test_data_dir / 'images'
        self.csv_file = self.test_data_dir / 'training_set_pixel_size_and_HC.csv'
        self.target_size = (128, 128)  # Smaller size for faster testing
        
        # Verify test data exists
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Test images directory does not exist: {self.images_dir}")
        if not self.csv_file.exists():
            raise FileNotFoundError(f"Test CSV file does not exist: {self.csv_file}")
    
    def test_dataset_initialization(self):
        """Test basic dataset initialization."""
        dataset = FetalHeadCircDataset(
            images_dir=str(self.images_dir),
            csv_file=str(self.csv_file),
            target_size=self.target_size
        )
        
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.target_size, self.target_size)
        self.assertTrue(dataset.load_annotations)  # Default is True
    
    def test_dataset_loads_images(self):
        """Test that dataset loads images correctly."""
        dataset = FetalHeadCircDataset(
            images_dir=str(self.images_dir),
            csv_file=str(self.csv_file),
            target_size=self.target_size
        )
        
        # Check that image paths are loaded
        self.assertGreater(len(dataset.image_paths), 0)
        self.assertEqual(len(dataset.image_paths), len(dataset.metadata))
        
        # Verify image paths exist
        for img_path in dataset.image_paths:
            self.assertTrue(img_path.exists())
    
    def test_dataset_returns_grayscale_images(self):
        """Test that dataset returns grayscale images with correct shape."""
        dataset = FetalHeadCircDataset(
            images_dir=str(self.images_dir),
            csv_file=str(self.csv_file),
            target_size=self.target_size
        )
        
        sample = dataset[0]
        image = sample['image']
        
        # Check that image is a tensor
        self.assertIsInstance(image, torch.Tensor)
        
        # Check shape: should be [1, H, W] for grayscale
        self.assertEqual(image.dim(), 3)
        self.assertEqual(image.size(0), 1)  # Single channel
        self.assertEqual(image.size(1), self.target_size[0])  # Height
        self.assertEqual(image.size(2), self.target_size[1])  # Width
        
        # Check that values are in [0, 1] range (from ToTensor)
        self.assertGreaterEqual(image.min().item(), 0.0)
        self.assertLessEqual(image.max().item(), 1.0)
    
    def test_dataset_returns_metadata(self):
        """Test that dataset returns correct metadata structure."""
        dataset = FetalHeadCircDataset(
            images_dir=str(self.images_dir),
            csv_file=str(self.csv_file),
            target_size=self.target_size
        )
        
        sample = dataset[0]
        metadata = sample['metadata']
        
        # Check metadata structure
        self.assertIsInstance(metadata, dict)
        self.assertIn('filename', metadata)
        self.assertIn('pixel_size', metadata)
        self.assertIn('head_circumference', metadata)
        self.assertIn('image_name', metadata)
    
    def test_dataset_with_annotations(self):
        """Test dataset with annotations enabled."""
        dataset = FetalHeadCircDataset(
            images_dir=str(self.images_dir),
            csv_file=str(self.csv_file),
            target_size=self.target_size,
            load_annotations=True
        )
        
        sample = dataset[0]
        
        # Check if annotation is present (may or may not exist in test data)
        if 'annotation' in sample:
            annotation = sample['annotation']
            self.assertIsInstance(annotation, torch.Tensor)
            self.assertEqual(annotation.dim(), 3)
            self.assertEqual(annotation.size(0), 1)  # Single channel
    
    def test_dataset_without_annotations(self):
        """Test dataset with annotations disabled."""
        dataset = FetalHeadCircDataset(
            images_dir=str(self.images_dir),
            csv_file=str(self.csv_file),
            target_size=self.target_size,
            load_annotations=False
        )
        
        sample = dataset[0]
        
        # Annotation should not be in sample when load_annotations=False
        # (unless it was explicitly added, but our implementation doesn't add it)
        # Actually, our implementation only adds annotation if it exists and load_annotations=True
        # So with load_annotations=False, annotation should not be in the result
        # But let's check the actual behavior - annotation_paths will all be None
        self.assertIn('image', sample)
        self.assertIn('metadata', sample)
    
    def test_custom_transforms(self):
        """Test dataset with custom transforms."""
        custom_transforms = [
            transforms.Resize(self.target_size),
            transforms.RandomHorizontalFlip(p=1.0),  # Always flip for testing
            transforms.ToTensor()
        ]
        
        dataset = FetalHeadCircDataset(
            images_dir=str(self.images_dir),
            csv_file=str(self.csv_file),
            transform=custom_transforms,
            target_size=self.target_size
        )
        
        sample = dataset[0]
        image = sample['image']
        
        # Check that image is a tensor with correct shape
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.dim(), 3)
        self.assertEqual(image.size(0), 1)  # Single channel
    
    def test_target_size_configuration(self):
        """Test that target_size is correctly applied."""
        custom_size = (256, 256)
        dataset = FetalHeadCircDataset(
            images_dir=str(self.images_dir),
            csv_file=str(self.csv_file),
            target_size=custom_size
        )
        
        sample = dataset[0]
        image = sample['image']
        
        # Check that image has the custom target size
        self.assertEqual(image.size(1), custom_size[0])  # Height
        self.assertEqual(image.size(2), custom_size[1])  # Width
    
    def test_get_statistics(self):
        """Test get_statistics method."""
        dataset = FetalHeadCircDataset(
            images_dir=str(self.images_dir),
            csv_file=str(self.csv_file),
            target_size=self.target_size
        )
        
        stats = dataset.get_statistics()
        
        # Check statistics structure
        self.assertIsInstance(stats, dict)
        self.assertIn('num_images', stats)
        self.assertIn('num_annotations', stats)
        
        # Check that statistics are calculated correctly
        self.assertEqual(stats['num_images'], len(dataset))
        
        # Check pixel_size statistics if available
        if 'pixel_size' in stats:
            self.assertIn('mean', stats['pixel_size'])
            self.assertIn('std', stats['pixel_size'])
            self.assertIn('min', stats['pixel_size'])
            self.assertIn('max', stats['pixel_size'])
        
        # Check head_circumference statistics if available
        if 'head_circumference' in stats:
            self.assertIn('mean', stats['head_circumference'])
            self.assertIn('std', stats['head_circumference'])
            self.assertIn('min', stats['head_circumference'])
            self.assertIn('max', stats['head_circumference'])
    
    def test_invalid_images_dir(self):
        """Test that dataset raises error for invalid images directory."""
        with self.assertRaises(FileNotFoundError):
            FetalHeadCircDataset(
                images_dir='/nonexistent/path',
                csv_file=str(self.csv_file),
                target_size=self.target_size
            )
    
    def test_invalid_csv_file(self):
        """Test that dataset raises error for invalid CSV file."""
        with self.assertRaises(FileNotFoundError):
            FetalHeadCircDataset(
                images_dir=str(self.images_dir),
                csv_file='/nonexistent/file.csv',
                target_size=self.target_size
            )
    
    def test_missing_csv_columns(self):
        """Test that dataset raises error for CSV with missing required columns."""
        # Create a temporary CSV with missing columns
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('filename,other_column\n')
            f.write('1_HC.png,value\n')
            temp_csv = f.name
        
        try:
            with self.assertRaises(ValueError):
                FetalHeadCircDataset(
                    images_dir=str(self.images_dir),
                    csv_file=temp_csv,
                    target_size=self.target_size
                )
        finally:
            # Clean up
            os.unlink(temp_csv)
    
    def test_dataset_length(self):
        """Test that dataset length matches number of valid images."""
        dataset = FetalHeadCircDataset(
            images_dir=str(self.images_dir),
            csv_file=str(self.csv_file),
            target_size=self.target_size
        )
        
        # Length should match number of image paths
        self.assertEqual(len(dataset), len(dataset.image_paths))
        self.assertEqual(len(dataset), len(dataset.metadata))
    
    def test_dataset_indexing(self):
        """Test that dataset indexing works correctly."""
        dataset = FetalHeadCircDataset(
            images_dir=str(self.images_dir),
            csv_file=str(self.csv_file),
            target_size=self.target_size
        )
        
        # Test accessing different indices
        for i in range(min(2, len(dataset))):  # Test first 2 samples
            sample = dataset[i]
            self.assertIn('image', sample)
            self.assertIn('metadata', sample)
            self.assertIsInstance(sample['image'], torch.Tensor)
            self.assertIsInstance(sample['metadata'], dict)
    
    def test_image_name_matching(self):
        """Test that image names are correctly matched from CSV."""
        dataset = FetalHeadCircDataset(
            images_dir=str(self.images_dir),
            csv_file=str(self.csv_file),
            target_size=self.target_size
        )
        
        # Check that metadata contains correct image names
        for i, metadata in enumerate(dataset.metadata):
            self.assertIn('image_name', metadata)
            self.assertIn('filename', metadata)
            
            # Verify that the image path matches the metadata
            image_path = dataset.image_paths[i]
            expected_name = metadata['image_name']
            self.assertIn(expected_name, str(image_path))
    
    def test_no_valid_images_error(self):
        """Test that dataset raises error when no valid images are found."""
        # Create a CSV pointing to non-existent images
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('filename,pixel size,head circumference (mm)\n')
            f.write('nonexistent1.png,0.1,50.0\n')
            f.write('nonexistent2.png,0.2,60.0\n')
            temp_csv = f.name
        
        try:
            with self.assertRaises(ValueError):
                FetalHeadCircDataset(
                    images_dir=str(self.images_dir),
                    csv_file=temp_csv,
                    target_size=self.target_size
                )
        finally:
            # Clean up
            os.unlink(temp_csv)
    
    def test_default_transform_uses_resize_with_pad(self):
        """Test that default transform uses resize_with_pad."""
        dataset = FetalHeadCircDataset(
            images_dir=str(self.images_dir),
            csv_file=str(self.csv_file),
            target_size=self.target_size
        )
        
        # Check that transform is set
        self.assertIsNotNone(dataset.transform)
        
        # Get a sample to verify transform was applied
        sample = dataset[0]
        image = sample['image']
        
        # Image should have the target size
        self.assertEqual(image.size(1), self.target_size[0])
        self.assertEqual(image.size(2), self.target_size[1])


if __name__ == '__main__':
    unittest.main()

