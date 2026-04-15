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

from embeddings.vit.train import resize_keep_aspect_pad


def composite_plane_label(plane: str, brain_plane: str) -> str:
    """
    Canonical string label for the pair (Plane, Brain_plane).

    Used as the classification key when ``class_to_idx`` is set and for
    ``get_class_counts`` with ``label_key='composite'``.
    """
    return f"{plane}::{brain_plane}"


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
        transform: List of transformation functions to apply (default: resize_keep_aspect_pad)
        target_size: Target size for resize (default: (224, 224), assumes square)
        csv_file: Name of the CSV file (default: 'FETAL_PLANES_DB_data.csv')
        images_dir: Name of images directory (default: 'images')
        train: If True, load training set; if False, load validation set; if None, load all (default: None)
        class_to_idx: Optional mapping from composite labels ``(Plane, Brain_plane)`` to indices
                      (see :func:`composite_plane_label`). If provided, ``__getitem__`` returns
                      ``label_idx`` and ``label_name`` is the composite string.
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
            self.df = self.df[self.df['Train '] == (1 if train else 0)]
        
        # Reset index after filtering
        self.df = self.df.reset_index(drop=True)
        
        # Verify images exist
        self.image_paths = []
        self.labels = []
        for idx, row in self.df.iterrows():
            image_name = row['Image_name']
            image_path = self.images_dir / f"{image_name}.png"
            
            if image_path.exists():
                plane = str(row.get('Plane', '') or '')
                brain_plane = str(row.get('Brain_plane', '') or '')
                comp = composite_plane_label(plane, brain_plane)

                # If class_to_idx is provided, filter out unknown composite classes
                if class_to_idx is not None and comp not in class_to_idx:
                    continue

                self.image_paths.append(image_path)
                # Store relevant metadata (composite = (Plane, Brain_plane) as one label)
                self.labels.append({
                    'Brain_plane': brain_plane,
                    'Plane': plane,
                    'composite': comp,
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
            # Default: resize_keep_aspect_pad (aspect ratio preserved, zero padding)
            self.transform = self._get_default_transforms()
        else:
            self.transform = transforms.Compose(transform) if transform else None
    
    def _get_default_transforms(self) -> transforms.Compose:
        """
        Get default transforms: resize with aspect ratio preserved and zero padding.

        Uses resize_keep_aspect_pad to fit within target_size while keeping proportions.
        """
        size = self.target_size[0]  # assumes square (H, W) e.g. (224, 224)

        def resize_pad_transform(img: torch.Tensor) -> torch.Tensor:
            return resize_keep_aspect_pad(
                img,
                size=size,
                interpolation=transforms.InterpolationMode.BILINEAR,
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
                - 'label': Dictionary with metadata (Brain_plane, Plane, composite, …)
                           OR ``label_idx`` / ``label_name`` if ``class_to_idx`` is provided
                              (``label_name`` is the composite ``Plane::Brain_plane`` string).
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
            # Classification mode: (Plane, Brain_plane) composite label
            comp = label['composite']
            label_idx = self.class_to_idx[comp]
            return {
                'image': image,
                'label_idx': torch.tensor(label_idx, dtype=torch.long),
                'label_name': comp,
            }
        else:
            # Default mode: return full label dict
            return {
                'image': image,
                'label': label
            }
    
    def get_class_counts(self, label_key: str = 'composite') -> dict:
        """
        Get counts of each class for a given label key.

        Args:
            label_key: Key in label dictionary to count. Default ``'composite'`` for
                       the ``(Plane, Brain_plane)`` pair; use ``'Brain_plane'`` or
                       ``'Plane'`` for marginal counts only.
        """
        counts = {}
        for label in self.labels:
            class_name = label.get(label_key, 'Unknown')
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

