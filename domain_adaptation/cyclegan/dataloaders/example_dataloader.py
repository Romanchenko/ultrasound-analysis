"""
Example dataloader implementation.
This serves as a template for creating custom dataloaders for specific datasets.
"""

from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional

from .base_dataloader import UnpairedDataset, create_dataloader


def create_cyclegan_dataloader(
    dataset_a: Dataset,
    dataset_b: Dataset,
    batch_size: int = 1,
    image_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    normalize_to_neg_one: bool = True
):
    """
    Create a dataloader for CycleGAN using two dataset instances.
    
    Args:
        dataset_a: Dataset instance for domain A (e.g., FetalHeadCircDataset)
        dataset_b: Dataset instance for domain B (e.g., FetalPlanesDBDataset)
        batch_size: Batch size
        image_size: Target image size (for additional resizing if needed)
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        normalize_to_neg_one: Whether to normalize images to [-1, 1] range.
                             Set to False if datasets already normalize.
    
    Returns:
        DataLoader instance
    """
    # Optional additional transforms for CycleGAN (normalize to [-1, 1])
    transform_a = None
    transform_b = None
    
    if normalize_to_neg_one:
        # Normalize to [-1, 1] range for CycleGAN
        # Assuming input is in [0, 1] range from ToTensor
        transform_a = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        transform_b = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        # Just resize if needed
        transform_a = transforms.Compose([
            transforms.Resize((image_size, image_size))
        ])
        transform_b = transforms.Compose([
            transforms.Resize((image_size, image_size))
        ])
    
    # Create unpaired dataset wrapper
    unpaired_dataset = UnpairedDataset(
        dataset_a=dataset_a,
        dataset_b=dataset_b,
        transform_a=transform_a,
        transform_b=transform_b
    )
    
    # Create dataloader
    return create_dataloader(
        dataset=unpaired_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

