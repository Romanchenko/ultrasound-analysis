"""
Visualization module for inspecting data from dataloaders.
Supports both DDTI-style and legacy dataloaders.

Designed for use in Jupyter notebooks. Import the functions and call them directly from notebook cells.

Example:
    from visualize_data import visualize_batch, print_dataloader_info
    from dataloader import create_ddti_dataloader
    
    dataloader = create_ddti_dataloader(root='./data/train', batch_size=4, num_workers=0)
    print_dataloader_info(dataloader)
    visualize_batch(dataloader, num_samples=8)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict, Tuple
from torch.utils.data import DataLoader
from pathlib import Path
import os

# Set matplotlib backend for notebooks (optional, but can help with display)
try:
    get_ipython()  # Check if running in IPython/Jupyter
    plt.ion()  # Turn on interactive mode
except NameError:
    pass  # Not in IPython, use default backend


def denormalize_image(
    image: torch.Tensor,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Denormalize an image tensor.
    
    Args:
        image: Normalized image tensor [C, H, W] or [B, C, H, W]
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        Denormalized image tensor
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    
    if image.dim() == 4:  # Batch dimension
        mean_tensor = mean_tensor.unsqueeze(0)
        std_tensor = std_tensor.unsqueeze(0)
    
    return image * std_tensor + mean_tensor


def tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    """
    Convert image tensor to numpy array for visualization.
    
    Args:
        image: Image tensor [C, H, W] or [H, W]
        
    Returns:
        Numpy array [H, W, C] or [H, W] ready for matplotlib
    """
    if image.dim() == 3:
        # [C, H, W] -> [H, W, C]
        image = image.permute(1, 2, 0)
    elif image.dim() == 2:
        # [H, W] -> [H, W]
        pass
    else:
        raise ValueError(f"Unexpected tensor shape: {image.shape}")
    
    # Convert to numpy and clip values
    img_np = image.detach().cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    
    # If grayscale (single channel), convert to 3 channels for display
    if img_np.shape[-1] == 1:
        img_np = np.repeat(img_np, 3, axis=-1)
    
    return img_np


def visualize_batch(
    dataloader: DataLoader,
    num_samples: int = 8,
    denorm_mean: Optional[List[float]] = None,
    denorm_std: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Visualize a batch of images from a dataloader.
    
    Args:
        dataloader: DataLoader to visualize
        num_samples: Number of samples to display
        denorm_mean: Mean values for denormalization (if images are normalized)
        denorm_std: Std values for denormalization (if images are normalized)
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    # Get a batch
    batch = next(iter(dataloader))
    
    # Handle both dict (DDTI) and tensor batches
    if isinstance(batch, dict):
        images = batch['image']
        is_dict_batch = True
    else:
        images = batch
        is_dict_batch = False
    
    # Limit number of samples
    num_samples = min(num_samples, images.size(0))
    images = images[:num_samples]
    
    # Denormalize if needed
    if denorm_mean is not None and denorm_std is not None:
        images = denormalize_image(images, denorm_mean, denorm_std)
    
    # Create figure
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        img = images[i]
        img_np = tensor_to_numpy(img)
        
        axes[i].imshow(img_np, cmap='gray' if img_np.shape[-1] == 1 else None)
        axes[i].axis('off')
        
        if is_dict_batch and 'image_name' in batch:
            axes[i].set_title(f"Sample {i+1}\n{batch['image_name'][i]}", fontsize=8)
        else:
            axes[i].set_title(f"Sample {i+1}", fontsize=10)
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()
        # Don't close in notebook mode to allow display
        # plt.close()  # Uncomment if you want to close figures automatically


def visualize_augmentations(
    dataloader: DataLoader,
    num_samples: int = 4,
    augment_transform: Optional[torch.nn.Module] = None,
    denorm_mean: Optional[List[float]] = None,
    denorm_std: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 10)
):
    """
    Visualize original and augmented versions of images.
    
    Args:
        dataloader: DataLoader to visualize
        num_samples: Number of samples to display
        augment_transform: Augmentation transform to apply
        denorm_mean: Mean values for denormalization
        denorm_std: Std values for denormalization
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    # Get a batch
    batch = next(iter(dataloader))
    
    # Handle both dict (DDTI) and tensor batches
    if isinstance(batch, dict):
        images = batch['image']
        is_dict_batch = True
    else:
        images = batch
        is_dict_batch = False
    
    # Limit number of samples
    num_samples = min(num_samples, images.size(0))
    images = images[:num_samples]
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        img = images[i]
        
        # Original image
        if denorm_mean is not None and denorm_std is not None:
            img_orig = denormalize_image(img.unsqueeze(0), denorm_mean, denorm_std).squeeze(0)
        else:
            img_orig = img
        
        img_orig_np = tensor_to_numpy(img_orig)
        axes[i, 0].imshow(img_orig_np, cmap='gray' if img_orig_np.shape[-1] == 1 else None)
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Original', fontsize=10)
        
        # Augmented image
        if augment_transform is not None:
            img_aug = augment_transform(img.unsqueeze(0)).squeeze(0)
            if denorm_mean is not None and denorm_std is not None:
                img_aug = denormalize_image(img_aug.unsqueeze(0), denorm_mean, denorm_std).squeeze(0)
            img_aug_np = tensor_to_numpy(img_aug)
            axes[i, 1].imshow(img_aug_np, cmap='gray' if img_aug_np.shape[-1] == 1 else None)
        else:
            axes[i, 1].text(0.5, 0.5, 'No augmentation\ntransform provided', 
                          ha='center', va='center', transform=axes[i, 1].transAxes)
        
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Augmented', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()
        # Don't close in notebook mode to allow display
        # plt.close()  # Uncomment if you want to close figures automatically


def visualize_dataset_statistics(
    dataloader: DataLoader,
    num_batches: int = 10,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Visualize dataset statistics (pixel value distributions).
    
    Args:
        dataloader: DataLoader to analyze
        num_batches: Number of batches to analyze
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    all_pixels = []
    image_shapes = []
    
    print(f"Analyzing {num_batches} batches...")
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        # Handle both dict (DDTI) and tensor batches
        if isinstance(batch, dict):
            images = batch['image']
        else:
            images = batch
        
        # Collect pixel values
        for img in images:
            # Use first channel only (all channels are same for grayscale)
            img_single = img[0] if img.dim() == 3 else img
            all_pixels.extend(img_single.flatten().cpu().numpy().tolist())
            image_shapes.append(img.shape)
    
    all_pixels = np.array(all_pixels)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Histogram of pixel values
    axes[0].hist(all_pixels, bins=100, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Pixel Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Pixel Value Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Statistics
    stats_text = f"""
    Statistics:
    Mean: {all_pixels.mean():.4f}
    Std: {all_pixels.std():.4f}
    Min: {all_pixels.min():.4f}
    Max: {all_pixels.max():.4f}
    Median: {np.median(all_pixels):.4f}
    """
    axes[1].text(0.1, 0.5, stats_text, fontsize=12, 
                verticalalignment='center', family='monospace')
    axes[1].axis('off')
    axes[1].set_title('Dataset Statistics')
    
    # Image shape distribution
    if image_shapes:
        shapes_str = [f"{s[1]}x{s[2]}" for s in image_shapes[:20]]  # Show first 20
        unique_shapes, counts = np.unique(shapes_str, return_counts=True)
        axes[2].bar(range(len(unique_shapes)), counts)
        axes[2].set_xticks(range(len(unique_shapes)))
        axes[2].set_xticklabels(unique_shapes, rotation=45, ha='right')
        axes[2].set_xlabel('Image Shape (HxW)')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Image Shape Distribution')
        axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()
        # Don't close in notebook mode to allow display
        # plt.close()  # Uncomment if you want to close figures automatically


def visualize_sample_images(
    dataloader: DataLoader,
    indices: Optional[List[int]] = None,
    num_samples: int = 16,
    denorm_mean: Optional[List[float]] = None,
    denorm_std: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 20)
):
    """
    Visualize specific images from the dataset by index.
    
    Args:
        dataloader: DataLoader to visualize
        indices: Specific indices to visualize (if None, shows first num_samples)
        num_samples: Number of samples to display (if indices not provided)
        denorm_mean: Mean values for denormalization
        denorm_std: Std values for denormalization
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    dataset = dataloader.dataset
    
    if indices is None:
        indices = list(range(min(num_samples, len(dataset))))
    
    # Create figure
    cols = 4
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if len(indices) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, ax in zip(indices, axes):
        if idx >= len(dataset):
            ax.axis('off')
            continue
        
        # Get sample
        sample = dataset[idx]
        
        # Handle both dict (DDTI) and tensor samples
        if isinstance(sample, dict):
            img = sample['image']
            img_name = sample.get('image_name', f'Image {idx}')
        else:
            img = sample
            img_name = f'Image {idx}'
        
        # Denormalize if needed
        if denorm_mean is not None and denorm_std is not None:
            img = denormalize_image(img.unsqueeze(0), denorm_mean, denorm_std).squeeze(0)
        
        img_np = tensor_to_numpy(img)
        ax.imshow(img_np, cmap='gray' if img_np.shape[-1] == 1 else None)
        ax.axis('off')
        ax.set_title(f"Index {idx}\n{img_name}", fontsize=8)
    
    # Hide unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()
        # Don't close in notebook mode to allow display
        # plt.close()  # Uncomment if you want to close figures automatically


def print_dataloader_info(dataloader: DataLoader):
    """
    Print information about a dataloader.
    
    Args:
        dataloader: DataLoader to inspect
    """
    dataset = dataloader.dataset
    
    print("=" * 60)
    print("DataLoader Information")
    print("=" * 60)
    print(f"Dataset type: {type(dataset).__name__}")
    print(f"Dataset length: {len(dataset)}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Number of workers: {dataloader.num_workers}")
    
    # Get a sample
    sample = dataset[0]
    if isinstance(sample, dict):
        print(f"Sample type: Dictionary")
        print(f"Sample keys: {list(sample.keys())}")
        if 'image' in sample:
            img = sample['image']
            print(f"Image shape: {img.shape}")
            print(f"Image dtype: {img.dtype}")
            if 'image_name' in sample:
                print(f"Image name: {sample['image_name']}")
    else:
        print(f"Sample type: Tensor")
        print(f"Image shape: {sample.shape}")
        print(f"Image dtype: {sample.dtype}")
    
    print("=" * 60)


# Jupyter notebook example usage:
#
# ```python
# from visualize_data import (
#     visualize_batch,
#     visualize_augmentations,
#     visualize_dataset_statistics,
#     visualize_sample_images,
#     print_dataloader_info
# )
# from dataloader import create_ddti_dataloader, create_dataloader, get_default_transform
#
# # Create dataloader (DDTI style)
# dataloader = create_ddti_dataloader(
#     root='/path/to/data/train',
#     batch_size=4,
#     shuffle=False,
#     num_workers=0  # Use 0 for visualization in notebooks
# )
#
# # Or legacy style
# # dataloader = create_dataloader(
# #     image_folders=['/path/to/images'],
# #     batch_size=4,
# #     shuffle=False,
# #     num_workers=0
# # )
#
# # Print dataloader information
# print_dataloader_info(dataloader)
#
# # Visualize a batch
# visualize_batch(dataloader, num_samples=8)
#
# # Visualize dataset statistics
# visualize_dataset_statistics(dataloader, num_batches=10)
#
# # Visualize specific samples
# visualize_sample_images(dataloader, indices=[0, 5, 10, 15])
#
# # Visualize augmentations (if you have an augmentation transform)
# # from train import create_augmentation_transform
# # aug_transform = create_augmentation_transform(image_size=224)
# # visualize_augmentations(dataloader, num_samples=4, augment_transform=aug_transform)
# ```

