"""
Base dataloader for CycleGAN.
Provides common functionality for creating paired/unpaired dataloaders.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Tuple


class UnpairedDataset(Dataset):
    """
    Wrapper class for unpaired image datasets.
    Takes two dataset instances (like those in datasets/ directory) and pairs them.
    """
    
    def __init__(
        self,
        dataset_a: Dataset,
        dataset_b: Dataset,
        transform_a: Optional[Callable] = None,
        transform_b: Optional[Callable] = None
    ):
        """
        Initialize unpaired dataset.
        
        Args:
            dataset_a: Dataset instance for domain A (e.g., FetalHeadCircDataset)
            dataset_b: Dataset instance for domain B (e.g., FetalPlanesDBDataset)
            transform_a: Optional additional transformations for domain A images.
                         Applied after dataset's own transforms.
            transform_b: Optional additional transformations for domain B images.
                         Applied after dataset's own transforms.
        """
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.transform_a = transform_a
        self.transform_b = transform_b
    
    def __len__(self) -> int:
        """Return the maximum length of both domains."""
        return max(len(self.dataset_a), len(self.dataset_b))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a pair of images from domains A and B.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image_a, image_b) as tensors
        """
        # Use modulo to cycle through images if one domain has fewer images
        idx_a = idx % len(self.dataset_a)
        idx_b = idx % len(self.dataset_b)
        
        # Get samples from datasets
        sample_a = self.dataset_a[idx_a]
        sample_b = self.dataset_b[idx_b]
        
        # Extract images from samples (handle both dict and tensor returns)
        if isinstance(sample_a, dict):
            img_a = sample_a['image']
        else:
            img_a = sample_a
        
        if isinstance(sample_b, dict):
            img_b = sample_b['image']
        else:
            img_b = sample_b
        
        # Apply additional transforms if provided
        if self.transform_a:
            img_a = self.transform_a(img_a)
        if self.transform_b:
            img_b = self.transform_b(img_b)
        
        # Ensure images are tensors
        if not isinstance(img_a, torch.Tensor):
            img_a = torch.tensor(img_a)
        if not isinstance(img_b, torch.Tensor):
            img_b = torch.tensor(img_b)
        
        return img_a, img_b


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader for CycleGAN training.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

