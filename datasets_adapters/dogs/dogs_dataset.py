"""
Dataset class for Dogs Photo dataset (four breeds).
Loads RGB images from breed-named subdirectories with configurable transformations.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Optional, Callable, Dict
from pathlib import Path
import torchvision.transforms as transforms
from torchtune.modules.transforms.vision_utils.resize_with_pad import resize_with_pad


# Breed folder names as they appear on disk
BREEDS = ['husky', 'labrador', 'golden retriever', 'german shepherd']


class DogsDataset(Dataset):
    """
    Dataset class for the Dogs Photo dataset (four breeds).

    Dataset structure:
        root/
            husky/
                image_1.jpg
                ...
            labrador/
                image_1.jpg
                ...
            golden retriever/
                image_1.jpg
                ...
            german shepherd/
                image_1.jpg
                ...

    Args:
        root: Root directory of the dataset.
        transform: Optional list of transformation functions to apply.
                   If None, default transforms (resize_with_pad) are used.
        target_size: Target size (H, W) for resize_with_pad (default: (224, 224)).
        breeds: List of breed folder names to include.
                If None, all four breeds are loaded.
        class_to_idx: Optional mapping from breed names to integer indices.
                      If None, one is built automatically from the loaded breeds.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[List[Callable]] = None,
        target_size: tuple = (224, 224),
        breeds: Optional[List[str]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.root = Path(root)
        self.target_size = target_size

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        # Determine which breeds to load
        self.breeds = breeds if breeds is not None else BREEDS

        # Build or store class mapping
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = {breed: idx for idx, breed in enumerate(sorted(self.breeds))}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Scan breed directories
        self.image_paths: List[Path] = []
        self.labels: List[str] = []

        for breed in self.breeds:
            breed_dir = self.root / breed
            if not breed_dir.exists():
                print(f"Warning: breed directory not found: {breed_dir}")
                continue

            for entry in sorted(breed_dir.iterdir()):
                if entry.is_file() and entry.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                    self.image_paths.append(entry)
                    self.labels.append(breed)

        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in {self.root}")

        print(f"Loaded {len(self.image_paths)} images across {len(set(self.labels))} breeds from {self.root}")

        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transforms.Compose(transform) if transform else None

    # ------------------------------------------------------------------
    # Default transforms
    # ------------------------------------------------------------------
    def _get_default_transforms(self) -> transforms.Compose:
        """Default: convert to tensor then resize_with_pad with zero (black) padding."""

        def resize_pad_transform(img):
            return resize_with_pad(
                img,
                target_size=self.target_size,
                resample=transforms.InterpolationMode.BILINEAR,
            )

        return transforms.Compose([
            transforms.ToTensor(),  # [3, H, W] for RGB
            transforms.Lambda(resize_pad_transform),
        ])

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Sample index.

        Returns:
            Dictionary with:
                - 'image':      image tensor [3, H, W] (RGB)
                - 'label_idx':  integer class index (torch.long)
                - 'label_name': breed name (str)
        """
        image_path = self.image_paths[idx]
        breed = self.labels[idx]

        # Load as RGB
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', self.target_size, color=(0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Ensure 3-channel tensor [3, H, W]
        if image.dim() == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        elif image.dim() == 3 and image.size(0) == 1:
            image = image.repeat(3, 1, 1)

        label_idx = self.class_to_idx[breed]

        return {
            'image': image,
            'label_idx': torch.tensor(label_idx, dtype=torch.long),
            'label_name': breed,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_class_counts(self) -> Dict[str, int]:
        """Return number of images per breed."""
        counts: Dict[str, int] = {}
        for breed in self.labels:
            counts[breed] = counts.get(breed, 0) + 1
        return counts

