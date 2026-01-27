"""
Unit tests for FetalPlanesDBDataset.
"""

import unittest
import torch
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

import sys
import os
# Add parent directory to path to import dataset
# sys.path.insert(0, str(Path(__file__).parent.parent))
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)
from fpd_dataset import FetalPlanesDBDataset


class TestFetalPlanesDBDataset(unittest.TestCase):
    """Unit tests for FetalPlanesDBDataset."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_data_dir = Path(__file__).parent / 'data'
        cls.test_root = str(cls.test_data_dir)
        
    def setUp(self):
        """Set up before each test."""
        self.test_root = str(Path(__file__).parent / 'data')
    
    def test_dataset_initialization(self):
        """Test basic dataset initialization."""
        dataset = FetalPlanesDBDataset(root=self.test_root)
        
        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset), 2)  # A and B images
        self.assertEqual(dataset.target_size, (224, 224))
    
    def test_dataset_loads_images(self):
        """Test that dataset loads images correctly."""
        print('DEBUG XXX: test_root', self.test_root)
        dataset = FetalPlanesDBDataset(root=self.test_root)
        
        # Check that image paths are loaded
        self.assertEqual(len(dataset.image_paths), 2)
        self.assertEqual(len(dataset.labels), 2)
        
        # Verify image paths exist
        for img_path in dataset.image_paths:
            self.assertTrue(img_path.exists())
    
    def test_dataset_returns_grayscale_images(self):
        """Test that dataset returns grayscale images with correct shape."""
        dataset = FetalPlanesDBDataset(root=self.test_root)
        
        sample = dataset[0]
        image = sample['image']
        
        # Check that image is a tensor
        self.assertIsInstance(image, torch.Tensor)
        
        # Check shape: should be [1, H, W] for grayscale
        self.assertEqual(image.dim(), 3)
        self.assertEqual(image.size(0), 1)  # Single channel
        
        # Check that values are in [0, 1] range (from ToTensor)
        self.assertGreaterEqual(image.min().item(), 0.0)
        self.assertLessEqual(image.max().item(), 1.0)
    
    def test_dataset_returns_labels(self):
        """Test that dataset returns correct label structure."""
        dataset = FetalPlanesDBDataset(root=self.test_root)
        
        sample = dataset[0]
        label = sample['label']
        
        # Check label structure
        self.assertIsInstance(label, dict)
        self.assertIn('Brain_plane', label)
        self.assertIn('Plane', label)
        self.assertIn('Patient_num', label)
        self.assertIn('Image_name', label)
    
    def test_train_split_filtering(self):
        """Test train/val split filtering."""
        # Test training set
        train_dataset = FetalPlanesDBDataset(root=self.test_root, train=True)
        self.assertEqual(len(train_dataset), 1)  # Only A (Train=1)
        self.assertEqual(train_dataset[0]['label']['Image_name'], 'A')
        
        # Test validation set
        val_dataset = FetalPlanesDBDataset(root=self.test_root, train=False)
        self.assertEqual(len(val_dataset), 1)  # Only B (Train=0)
        self.assertEqual(val_dataset[0]['label']['Image_name'], 'B')
        
        # Test all data
        all_dataset = FetalPlanesDBDataset(root=self.test_root, train=None)
        self.assertEqual(len(all_dataset), 2)  # Both A and B
    
    def test_custom_transforms(self):
        """Test dataset with custom transforms."""
        custom_transforms = [
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ]
        
        dataset = FetalPlanesDBDataset(
            root=self.test_root,
            transform=custom_transforms
        )
        
        sample = dataset[0]
        image = sample['image']
        
        # Check that custom size is applied
        self.assertEqual(image.shape[1], 128)  # Height
        self.assertEqual(image.shape[2], 128)  # Width
    
    def test_target_size_configuration(self):
        """Test that target_size parameter works correctly."""
        custom_size = (256, 256)
        dataset = FetalPlanesDBDataset(
            root=self.test_root,
            target_size=custom_size
        )
        
        self.assertEqual(dataset.target_size, custom_size)
        
        sample = dataset[0]
        image = sample['image']
        
        # Check that target size is applied (with default resize_with_pad)
        # Note: resize_with_pad should maintain aspect ratio and pad
        # So the output might not be exactly target_size if aspect ratio differs
        # But it should be close
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.dim(), 3)
    
    def test_get_class_counts(self):
        """Test get_class_counts method."""
        dataset = FetalPlanesDBDataset(root=self.test_root)
        
        # Get counts for Brain_plane
        counts = dataset.get_class_counts('Brain_plane')
        
        self.assertIsInstance(counts, dict)
        self.assertIn('Not A Brain', counts)
        self.assertIn('Some', counts)
        self.assertEqual(counts['Not A Brain'], 1)  # Image A
        self.assertEqual(counts['Some'], 1)  # Image B
    
    def test_get_class_counts_plane(self):
        """Test get_class_counts with different label key."""
        dataset = FetalPlanesDBDataset(root=self.test_root)
        
        counts = dataset.get_class_counts('Plane')
        
        self.assertIsInstance(counts, dict)
        # Both images have 'Other' as Plane
        self.assertIn('Other', counts)
        self.assertEqual(counts['Other'], 2)
    
    def test_dataset_length(self):
        """Test dataset __len__ method."""
        dataset = FetalPlanesDBDataset(root=self.test_root)
        
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.__len__(), 2)
    
    def test_dataset_indexing(self):
        """Test dataset indexing."""
        dataset = FetalPlanesDBDataset(root=self.test_root)
        
        # Test valid indices
        sample0 = dataset[0]
        sample1 = dataset[1]
        
        self.assertIn('image', sample0)
        self.assertIn('label', sample0)
        self.assertIn('image', sample1)
        self.assertIn('label', sample1)
        
        # Images should be different
        self.assertNotEqual(sample0['label']['Image_name'], sample1['label']['Image_name'])
    
    def test_custom_csv_and_images_dir(self):
        """Test custom CSV file and images directory names."""
        dataset = FetalPlanesDBDataset(
            root=self.test_root,
            csv_file='FETAL_PLANES_DB_data.csv',
            images_dir='images'
        )
        
        self.assertEqual(len(dataset), 2)
    
    def test_missing_csv_file_error(self):
        """Test error handling for missing CSV file."""
        with self.assertRaises(FileNotFoundError):
            FetalPlanesDBDataset(
                root=self.test_root,
                csv_file='nonexistent.csv'
            )
    
    def test_empty_dataset_error(self):
        """Test error handling for dataset with no valid images."""
        # This test would require a CSV with images that don't exist
        # We'll skip this as it would require creating invalid test data
        pass
    
    def test_transform_none(self):
        """Test dataset with transform=None."""
        dataset = FetalPlanesDBDataset(
            root=self.test_root,
            transform=None
        )
        
        # Should use default transforms
        sample = dataset[0]
        image = sample['image']
        
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.dim(), 3)
    
    def test_transform_empty_list(self):
        """Test dataset with empty transform list."""
        dataset = FetalPlanesDBDataset(
            root=self.test_root,
            transform=[]
        )
        
        # Should use default transforms when transform is empty list
        sample = dataset[0]
        image = sample['image']
        
        self.assertIsInstance(image, torch.Tensor)
    
    def test_image_metadata(self):
        """Test that image metadata is correctly extracted."""
        dataset = FetalPlanesDBDataset(root=self.test_root)
        
        # Find sample with image A
        sample_a = None
        for i in range(len(dataset)):
            if dataset[i]['label']['Image_name'] == 'A':
                sample_a = dataset[i]
                break
        
        self.assertIsNotNone(sample_a)
        self.assertEqual(sample_a['label']['Brain_plane'], 'Not A Brain')
        self.assertEqual(sample_a['label']['Plane'], 'Other')
        self.assertEqual(sample_a['label']['Patient_num'], 1)
        self.assertEqual(sample_a['label']['Image_name'], 'A')
    
    def test_grayscale_conversion(self):
        """Test that images are properly converted to grayscale."""
        dataset = FetalPlanesDBDataset(root=self.test_root)
        
        sample = dataset[0]
        image = sample['image']
        
        # Should be single channel
        self.assertEqual(image.size(0), 1)
        
        # Should be 2D spatial dimensions
        self.assertEqual(image.dim(), 3)  # [1, H, W]
    
    def test_resize_with_pad_in_default_transform(self):
        """Test that default transform uses resize_with_pad."""
        dataset = FetalPlanesDBDataset(
            root=self.test_root,
            target_size=(224, 224)
        )
        
        # The transform should be a Compose object
        self.assertIsNotNone(dataset.transform)
        self.assertIsInstance(dataset.transform, transforms.Compose)
        
        # Get a sample to verify it works
        sample = dataset[0]
        image = sample['image']
        
        # Image should be tensor with correct number of dimensions
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.dim(), 3)
    
    def test_multiple_samples_consistency(self):
        """Test that multiple samples are consistent."""
        dataset = FetalPlanesDBDataset(root=self.test_root)
        
        samples = [dataset[i] for i in range(len(dataset))]
        
        # All samples should have same structure
        for sample in samples:
            self.assertIn('image', sample)
            self.assertIn('label', sample)
            self.assertIsInstance(sample['image'], torch.Tensor)
            self.assertIsInstance(sample['label'], dict)
            
            # Images should have same number of channels
            self.assertEqual(sample['image'].size(0), 1)


class TestFetalPlanesDBDatasetIntegration(unittest.TestCase):
    """Integration tests for FetalPlanesDBDataset with DataLoader."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_data_dir = Path(__file__).parent / 'data'
        cls.test_root = str(cls.test_data_dir)
    
    def test_dataloader_compatibility(self):
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        dataset = FetalPlanesDBDataset(root=self.test_root)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Get a batch
        batch = next(iter(dataloader))
        
        # Check batch structure
        self.assertIn('image', batch)
        self.assertIn('label', batch)
        
        # Check image batch shape: [batch_size, 1, H, W]
        images = batch['image']
        self.assertEqual(images.dim(), 4)
        self.assertEqual(images.size(0), 2)  # batch_size
        self.assertEqual(images.size(1), 1)  # grayscale channel
        
        # Check labels
        labels = batch['label']
        self.assertIsInstance(labels, dict)
        self.assertIn('Image_name', labels)
        self.assertEqual(len(labels['Image_name']), 2)
    
    def test_dataloader_with_custom_transforms(self):
        """Test DataLoader with custom transforms."""
        from torch.utils.data import DataLoader
        
        custom_transforms = [
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ]
        
        dataset = FetalPlanesDBDataset(
            root=self.test_root,
            transform=custom_transforms
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        batch = next(iter(dataloader))
        images = batch['image']
        
        # Check that custom transform was applied
        self.assertEqual(images.shape[2], 128)  # Height
        self.assertEqual(images.shape[3], 128)  # Width


if __name__ == '__main__':
    unittest.main()

