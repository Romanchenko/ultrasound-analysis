"""
Dataset class for FETAL_PLANES_DB dataset.
Loads grayscale ultrasound images with configurable transformations.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Optional, Callable, Any, Dict
from pathlib import Path
import torchvision.transforms as transforms
from torchtune.modules.transforms.vision_utils.resize_with_pad import resize_with_pad


class FetalPlanesDBDataset(Dataset):
    """
    Dataset class for FETAL_PLANES_DB grayscale ultrasound images.
    
    Dataset structure:
        root/
            FETAL_PLANES_DB_data.csv
            images/
                <image_name>.png
    
    Args:
        root: Root directory of the dataset
        transform: List of transformation functions to apply (default: resize_with_pad)
        target_size: Target size for resize_with_pad (default: (224, 224))
        csv_file: Name of the CSV file (default: 'FETAL_PLANES_DB_data.csv')
        images_dir: Name of images directory (default: 'images')
        train: If True, load training set; if False, load validation set; if None, load all (default: None)
        class_to_idx: Optional mapping from Brain_plane class names to indices for classification.
                      If provided, __getitem__ will return 'label_idx' instead of 'label' dict.
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[List[Callable]] = None,
        target_size: tuple = (224, 224),
        csv_file: str = 'FETAL_PLANES_DB_data.csv',
        images_dir: str = 'images',
        train: Optional[bool] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.root = Path(root)
        self.images_dir = self.root / images_dir
        self.csv_path = self.root / csv_file
        self.target_size = target_size
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx else None
        
        # Load CSV file
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        # Read CSV with semicolon separator
        self.df = pd.read_csv(self.csv_path, sep=';')
        
        # Filter by train/val split if specified
        if train is not None:
            self.df = self.df[self.df['Train'] == (1 if train else 0)]
        
        # Reset index after filtering
        self.df = self.df.reset_index(drop=True)
        
        # Verify images exist
        self.image_paths = []
        self.labels = []
        for idx, row in self.df.iterrows():
            image_name = row['Image_name']
            image_path = self.images_dir / f"{image_name}.png"
            
            if image_path.exists():
                brain_plane = row.get('Brain_plane', '')
                
                # If class_to_idx is provided, filter out unknown classes
                if class_to_idx is not None and brain_plane not in class_to_idx:
                    continue
                
                self.image_paths.append(image_path)
                # Store relevant metadata
                self.labels.append({
                    'Brain_plane': brain_plane,
                    'Plane': row.get('Plane', ''),
                    'Patient_num': row.get('Patient_num', ''),
                    'Image_name': image_name
                })
            else:
                print(f"Warning: Image not found: {image_path}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in {self.images_dir}")
        
        print(f"Loaded {len(self.image_paths)} images from {self.root}")
        
        # Set up transforms
        if transform is None:
            # Default: resize_with_pad with zero padding
            self.transform = self._get_default_transforms()
        else:
            self.transform = transforms.Compose(transform) if transform else None
    
    def _get_default_transforms(self) -> transforms.Compose:
        """
        Get default transforms including resize_with_pad with zero padding.
        
        Returns:
            Compose transform with resize_with_pad
        """
        
        # Create resize_with_pad transform with zero padding
        # resize_with_pad typically takes target_size as (height, width) tuple
        def resize_pad_transform(img):
            """Apply resize_with_pad with zero (black) padding."""
            # resize_with_pad expects PIL Image and returns PIL Image
            # padding_value=0 means black padding
            # target_size should be (height, width)
            return resize_with_pad(
                img,
                target_size=self.target_size,  # (H, W)
                resample=transforms.InterpolationMode.BILINEAR,
                # padding_value=0  # Black padding (zero values)
            )
        
        return transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor [1, H, W] for grayscale
            transforms.Lambda(resize_pad_transform),
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary with:
                - 'image': Image tensor [1, H, W] (grayscale)
                - 'label': Dictionary with metadata (Brain_plane, Plane, Patient_num, Image_name)
                           OR 'label_idx': Integer label index (if class_to_idx is provided)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image as grayscale
        try:
            image = Image.open(image_path).convert('L')  # 'L' mode for grayscale
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('L', self.target_size, color=0)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: just convert to tensor
            image = transforms.ToTensor()(image)
        
        # Ensure image is single channel tensor [1, H, W]
        if image.dim() == 2:
            image = image.unsqueeze(0)  # Add channel dimension
        elif image.dim() == 3 and image.size(0) > 1:
            # If multiple channels, take first channel (shouldn't happen for grayscale)
            image = image[0:1]
        
        # Return format depends on whether class_to_idx is provided
        if self.class_to_idx is not None:
            # Classification mode: return label_idx
            brain_plane = label['Brain_plane']
            label_idx = self.class_to_idx[brain_plane]
            return {
                'image': image,
                'label_idx': torch.tensor(label_idx, dtype=torch.long),
                'label_name': brain_plane
            }
        else:
            # Default mode: return full label dict
            return {
                'image': image,
                'label': label
            }
    
    def get_class_counts(self, label_key: str = 'Brain_plane') -> dict:
        """
        Get counts of each class for a given label key.
        
        Args:
            label_key: Key in label dictionary to count (default: 'Brain_plane')
            
        Returns:
            Dictionary mapping class names to counts
        """
        counts = {}
        for label in self.labels:
            class_name = label.get(label_key, 'Unknown')
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

