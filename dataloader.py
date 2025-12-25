"""
DataLoader module for loading ultrasound images from folders.
Based on DDTI dataloader structure from TRFE-Net.
Supports loading images from DDTI-style directory structure for training and testing.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np


class DDTI(data.Dataset):
    """
    DDTI-style dataset for loading ultrasound images and masks for nodule segmentation.
    Based on the DDTI dataloader from TRFE-Net for thyroid nodule segmentation.
    
    Expected directory structure:
        root/
            image/
                *.jpg, *.png, etc.
            masks/
                *.jpg, *.png, etc. (same filenames as images)
    
    Args:
        root: Root directory containing 'image' and 'masks' subdirectories
        transform: Optional transform to apply to images and masks (works on dict with 'image' and 'mask' keys)
        return_size: Whether to return image size in the sample
    """
    
    def __init__(self, root: str, transform=None, return_size: bool = False):
        self.root = root
        self.transform = transform
        self.return_size = return_size
        
        # Get image directory
        image_dir = os.path.join(root, 'image')
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory does not exist: {image_dir}")
        
        # Get masks directory
        masks_dir = os.path.join(root, 'mask')
        if not os.path.exists(masks_dir):
            raise ValueError(f"Masks directory does not exist: {masks_dir}")
        
        # Get all image files
        img_names = os.listdir(image_dir)
        # Sort by numeric value if filenames are numeric
        try:
            img_names = sorted(img_names, key=lambda i: int(i.split(".")[0]))
        except ValueError:
            # If not numeric, use alphabetical sort
            img_names = sorted(img_names)
        
        # Filter to only include images that have corresponding masks
        valid_img_names = []
        for img_name in img_names:
            mask_path = os.path.join(masks_dir, img_name)
            if os.path.exists(mask_path):
                valid_img_names.append(img_name)
            else:
                print(f"Warning: No mask found for {img_name}, skipping")
        
        self.img_names = valid_img_names
        
        if len(self.img_names) == 0:
            raise ValueError(f"No valid image-mask pairs found in {image_dir} and {masks_dir}")
        
        print(f"Found {len(self.img_names)} image-mask pairs")
    
    def __getitem__(self, index: int) -> Dict:
        """
        Get an image and mask by index.
        
        Args:
            index: Index of the image
            
        Returns:
            Dictionary with 'image' and 'mask' keys containing transformed tensors
            and optionally 'size' and 'image_name' keys
        """
        img_name = self.img_names[index]
        image_path = os.path.join(self.root, 'image', img_name)
        mask_path = os.path.join(self.root, 'mask', img_name)
        
        assert os.path.exists(image_path), f'{image_path} does not exist'
        assert os.path.exists(mask_path), f'{mask_path} does not exist'
        
        # Load image as RGB
        image = Image.open(image_path).convert('RGB')
        
        # Load mask as grayscale (L mode)
        mask = Image.open(mask_path).convert('L')
        
        # Store original size if needed
        if self.return_size:
            w, h = image.size
            size = (h, w)
        
        # Create sample dictionary (DDTI-style)
        sample = {'image': image, 'mask': mask}
        
        # Apply transform if provided (transform should work on dict)
        if self.transform:
            sample = self.transform(sample)
        else:
            # Default: convert to tensor
            to_tensor = transforms.ToTensor()
            image_tensor = to_tensor(sample['image'])  # [3, H, W] for RGB
            mask_tensor = to_tensor(sample['mask'])  # [1, H, W] for grayscale
            sample['image'] = image_tensor
            sample['mask'] = mask_tensor
        
        # Ensure image is 3-channel tensor (should already be RGB)
        if isinstance(sample['image'], torch.Tensor):
            if sample['image'].dim() == 3 and sample['image'].size(0) != 3:
                # If somehow not 3 channels, convert to RGB
                if sample['image'].size(0) == 1:
                    sample['image'] = sample['image'].repeat(3, 1, 1)
                else:
                    raise ValueError(f"Unexpected number of channels: {sample['image'].size(0)}")
        
        # After transform, ensure all values are tensors or other collatable types
        # Check if transform returned only image (for mean/std calculation)
        if 'mask' not in sample:
            # Transform already cleaned up the sample (e.g., for mean/std calculation)
            # Just ensure image is a tensor
            if not isinstance(sample['image'], torch.Tensor):
                to_tensor = transforms.ToTensor()
                sample['image'] = to_tensor(sample['image'])
        else:
            # Normal case: ensure mask is single channel (only if mask is present and is a tensor)
            if isinstance(sample['mask'], torch.Tensor):
                if sample['mask'].dim() == 3 and sample['mask'].size(0) > 1:
                    # Take first channel if multiple channels
                    sample['mask'] = sample['mask'][0:1]
            elif isinstance(sample['mask'], Image.Image):
                # Convert mask to tensor if it's still a PIL Image
                to_tensor = transforms.ToTensor()
                mask_tensor = to_tensor(sample['mask'])
                sample['mask'] = mask_tensor
        
        # Only add non-tensor fields if they're needed and won't cause collation issues
        if self.return_size:
            sample['size'] = torch.tensor(size)
        
        sample['image_name'] = img_name
        
        return sample
    
    def __len__(self) -> int:
        return len(self.img_names)


class UltrasoundImageDataset(Dataset):
    """
    Dataset class for loading ultrasound images from folders.
    
    Args:
        image_folders: List of folder paths containing images
        transform: Optional transform to apply to images
        image_extensions: List of image file extensions to include
    """
    
    def __init__(
        self,
        image_folders: List[str],
        transform: Optional[transforms.Compose] = None,
        image_extensions: Optional[List[str]] = None
    ):
        self.image_folders = [Path(folder) for folder in image_folders]
        self.transform = transform
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Collect all image paths
        self.image_paths = []
        for folder in self.image_folders:
            if not folder.exists():
                raise ValueError(f"Folder does not exist: {folder}")
            
            # Recursively find all images in the folder
            for ext in self.image_extensions:
                self.image_paths.extend(folder.rglob(f"*{ext}"))
                self.image_paths.extend(folder.rglob(f"*{ext.upper()}"))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in the specified folders: {image_folders}")
        
        print(f"Found {len(self.image_paths)} images in {len(self.image_folders)} folder(s)")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get an image by index.
        
        Args:
            idx: Index of the image
            
        Returns:
            Transformed image tensor [3, H, W] (RGB images)
        """
        image_path = self.image_paths[idx]
        
        try:
            # Load image as RGB
            image = Image.open(image_path).convert('RGB')
            
            # Apply transform if provided
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform: convert to tensor
                to_tensor = transforms.ToTensor()
                image = to_tensor(image)  # [3, H, W] for RGB
            
            # Ensure image is 3-channel tensor (should already be RGB)
            if image.dim() == 3 and image.size(0) != 3:
                if image.size(0) == 1:
                    # If single channel, replicate to 3 channels
                    image = image.repeat(3, 1, 1)
                else:
                    raise ValueError(f"Unexpected number of channels: {image.size(0)}")
            
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                fallback_img = Image.new('RGB', (224, 224), color='black')
                image = self.transform(fallback_img)
                if image.dim() == 3 and image.size(0) != 3:
                    if image.size(0) == 1:
                        image = image.repeat(3, 1, 1)
                return image
            else:
                return torch.zeros(3, 224, 224)


def create_ddti_dataloader(
    root: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None,
    return_size: bool = False
) -> DataLoader:
    """
    Create a DataLoader using DDTI-style dataset structure.
    
    Args:
        root: Root directory containing 'image' subdirectory
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        transform: Optional transform to apply to images (works on dict with 'image' key)
        return_size: Whether to return image size in samples
        
    Returns:
        DataLoader instance
    """
    dataset = DDTI(root=root, transform=transform, return_size=return_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


def create_dataloader(
    image_folders: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[transforms.Compose] = None,
    image_extensions: Optional[List[str]] = None
) -> DataLoader:
    """
    Create a DataLoader for ultrasound images.
    
    Args:
        image_folders: List of folder paths containing images
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        transform: Optional transform to apply to images
        image_extensions: List of image file extensions to include
        
    Returns:
        DataLoader instance
    """
    dataset = UltrasoundImageDataset(
        image_folders=image_folders,
        transform=transform,
        image_extensions=image_extensions
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


def calculate_mean_std_ddti(
    root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    num_samples: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """
    Calculate mean and standard deviation for normalization from DDTI-style dataset.
    Works with RGB images (3 channels).
    
    Args:
        root: Root directory containing 'image' subdirectory
        image_size: Target image size for resizing
        batch_size: Batch size for processing
        num_workers: Number of worker processes
        num_samples: Optional limit on number of samples to use (None = use all)
        
    Returns:
        Tuple of (mean, std) as lists of 3 values (one per RGB channel)
    """
    # Create a transform that only resizes and converts to tensor (no normalization)
    def simple_transform(sample):
        """Simple transform for mean/std calculation."""
        img = sample['image']
        # Ensure it's a PIL Image
        if not isinstance(img, Image.Image):
            if isinstance(img, torch.Tensor):
                # Already a tensor, just resize if needed
                if img.shape[-1] != image_size or img.shape[-2] != image_size:
                    img = F.interpolate(img.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0)
                img_tensor = img
            else:
                img = Image.fromarray(np.array(img))
                transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor()  # Converts RGB to [3, H, W]
                ])
                img_tensor = transform(img)
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()  # Converts RGB to [3, H, W]
            ])
            img_tensor = transform(img)
        
        # Return only image tensor (no mask needed for statistics)
        return {'image': img_tensor}
    
    # Create dataset
    dataset = DDTI(root=root, transform=simple_transform, return_size=False)
    
    # Limit number of samples if specified
    if num_samples is not None and num_samples < len(dataset):
        dataset.img_names = dataset.img_names[:num_samples]
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # For RGB images, calculate mean and std for each channel
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    print("Calculating mean and std from DDTI dataset (RGB images)...")
    for batch in tqdm(dataloader, desc="Computing statistics"):
        # batch is a dict, get 'image' key
        images = batch['image']  # [batch_size, 3, H, W]
        
        # Calculate mean and std for each channel
        images = images.view(images.size(0), images.size(1), -1)  # [batch_size, 3, H*W]
        
        # Mean and std over spatial dimensions (H*W), then sum over batch
        mean += images.mean(2).sum(0)  # [3]
        std += images.std(2).sum(0)    # [3]
        
        total_samples += images.size(0)
    
    # Average over all samples
    mean /= total_samples
    std /= total_samples
    
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    print(f"Calculated mean (RGB): {mean_list}")
    print(f"Calculated std (RGB): {std_list}")
    
    return mean_list, std_list


def calculate_mean_std(
    image_folders: List[str],
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    num_samples: Optional[int] = None,
    image_extensions: Optional[List[str]] = None
) -> Tuple[List[float], List[float]]:
    """
    Calculate mean and standard deviation for normalization from image folders.
    Works with RGB images (3 channels).
    
    Args:
        image_folders: List of folder paths containing images
        image_size: Target image size for resizing
        batch_size: Batch size for processing
        num_workers: Number of worker processes
        num_samples: Optional limit on number of samples to use (None = use all)
        image_extensions: List of image file extensions to include
        
    Returns:
        Tuple of (mean, std) as lists of 3 values (one per RGB channel)
    """
    # Create a transform that only resizes and converts to tensor (no normalization)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()  # Converts RGB to [3, H, W]
    ])
    
    # Create dataset without normalization
    dataset = UltrasoundImageDataset(
        image_folders=image_folders,
        transform=transform,
        image_extensions=image_extensions
    )
    
    # Limit number of samples if specified
    if num_samples is not None and num_samples < len(dataset):
        dataset.image_paths = dataset.image_paths[:num_samples]
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # For RGB images, calculate mean and std for each channel
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    print("Calculating mean and std from dataset (RGB images)...")
    for batch in tqdm(dataloader, desc="Computing statistics"):
        # batch shape: [batch_size, 3, H, W] for RGB
        batch = batch.view(batch.size(0), batch.size(1), -1)  # [batch_size, 3, H*W]
        
        # Calculate mean and std for each channel
        # Mean over spatial dimensions (H*W), then sum over batch
        mean += batch.mean(2).sum(0)  # [3]
        # Std over spatial dimensions (H*W), then sum over batch
        std += batch.std(2).sum(0)    # [3]
        
        total_samples += batch.size(0)
    
    # Average over all samples
    mean /= total_samples
    std /= total_samples
    
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    print(f"Calculated mean (RGB): {mean_list}")
    print(f"Calculated std (RGB): {std_list}")
    
    return mean_list, std_list


def get_default_transform(
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> transforms.Compose:
    """
    Get default transform for RGB images (resize, convert to tensor, normalize).
    
    Args:
        image_size: Target image size
        mean: Mean values for normalization (3 values for RGB channels). If None, uses ImageNet defaults.
        std: Std values for normalization (3 values for RGB channels). If None, uses ImageNet defaults.
        
    Returns:
        Compose transform for RGB images
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet defaults
    if std is None:
        std = [0.229, 0.224, 0.225]   # ImageNet defaults
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # Converts RGB PIL image to [3, H, W] tensor
        transforms.Normalize(mean=mean, std=std)
    ])

