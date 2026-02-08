"""
FID (Frechet Inception Distance) calculator using ResNet-50 embeddings.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from typing import Optional
import numpy as np
from tqdm import tqdm
from scipy import linalg


def get_resnet50_features_model(device: Optional[torch.device] = None):
    """
    Get ResNet-50 model for feature extraction.
    
    Args:
        device: Device to load model on
        
    Returns:
        ResNet-50 model with classification head removed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained ResNet-50
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # Remove classification head, keep only feature extractor
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model = model.to(device)
    
    return model


def prepare_image_transform(image_size: int = 224):
    """
    Create transform to prepare images for ResNet-50.
    Handles grayscale images by converting to RGB.
    
    Args:
        image_size: Target image size (default: 224 for ResNet-50)
        
    Returns:
        Transform function
    """
    def transform_fn(sample):
        """
        Transform function that handles both dict and tensor inputs.
        
        Args:
            sample: Either a dict with 'image' key or a tensor
            
        Returns:
            Transformed image tensor [3, H, W]
        """
        # Extract image from sample
        if isinstance(sample, dict):
            image = sample['image']
        else:
            image = sample
        
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        
        # Handle grayscale images (1 channel) -> convert to RGB (3 channels)
        if image.dim() == 2:
            # 2D tensor [H, W] -> add channel dimension
            image = image.unsqueeze(0)
        
        if image.dim() == 3 and image.size(0) == 1:
            # Single channel [1, H, W] -> replicate to RGB [3, H, W]
            image = image.repeat(3, 1, 1)
        elif image.dim() == 3 and image.size(0) > 3:
            # More than 3 channels, take first 3
            image = image[:3]
        
        # Resize to target size if needed
        if image.size(1) != image_size or image.size(2) != image_size:
            resize_transform = transforms.Resize((image_size, image_size))
            image = resize_transform(image)
        
        # Normalize using ImageNet statistics
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        
        return image
    
    return transform_fn


def extract_features(
    dataset: Dataset,
    model: Optional[nn.Module] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    image_size: int = 224
) -> np.ndarray:
    """
    Extract features from dataset using ResNet-50.
    
    Args:
        dataset: PyTorch Dataset instance
        model: Optional pretrained model (if None, ResNet-50 will be loaded)
        batch_size: Batch size for feature extraction
        num_workers: Number of worker processes
        device: Device to run on (auto-detected if None)
        image_size: Target image size for ResNet-50
        
    Returns:
        Numpy array of features [num_samples, feature_dim]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load model if not provided
    if model is None:
        model = get_resnet50_features_model(device)
    
    # Create transform
    transform_fn = prepare_image_transform(image_size)
    
    # Create wrapper dataset that applies transform
    class TransformedDataset(Dataset):
        def __init__(self, base_dataset, transform_fn):
            self.base_dataset = base_dataset
            self.transform_fn = transform_fn
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            sample = self.base_dataset[idx]
            return self.transform_fn(sample)
    
    transformed_dataset = TransformedDataset(dataset, transform_fn)
    
    # Create dataloader
    dataloader = DataLoader(
        transformed_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Extract features
    features_list = []
    
    print("Extracting features...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            batch = batch.to(device)
            
            # Forward pass
            features = model(batch)
            
            # Flatten features: [batch_size, channels, H, W] -> [batch_size, features]
            features = features.view(features.size(0), -1)
            
            features_list.append(features.cpu().numpy())
    
    # Concatenate all features
    all_features = np.concatenate(features_list, axis=0)
    
    print(f"Extracted {all_features.shape[0]} features of dimension {all_features.shape[1]}")
    
    return all_features


def calculate_fid(
    features1: np.ndarray,
    features2: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Calculate Frechet Inception Distance between two sets of features.
    
    Args:
        features1: Features from first dataset [num_samples1, feature_dim]
        features2: Features from second dataset [num_samples2, feature_dim]
        eps: Small value to add for numerical stability
        
    Returns:
        FID score (lower is better, 0 means identical distributions)
    """
    # Calculate mean and covariance for each set
    mu1 = np.mean(features1, axis=0)
    sigma1 = np.cov(features1, rowvar=False)
    
    mu2 = np.mean(features2, axis=0)
    sigma2 = np.cov(features2, rowvar=False)
    
    # Calculate FID
    # FID = ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*sqrt(C1*C2))
    
    # Squared difference of means
    diff = mu1 - mu2
    mean_diff_sq = np.sum(diff ** 2)
    
    # Covariance term
    # sqrt(C1*C2) using matrix square root
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    
    # Check for complex numbers (shouldn't happen with valid covariances)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate trace
    tr_covmean = np.trace(covmean)
    
    # FID score
    fid = mean_diff_sq + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    return float(fid)


def calculate_fid_from_datasets(
    dataset1: Dataset,
    dataset2: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    image_size: int = 224
) -> float:
    """
    Calculate FID between two datasets.
    
    Args:
        dataset1: First PyTorch Dataset
        dataset2: Second PyTorch Dataset
        batch_size: Batch size for feature extraction
        num_workers: Number of worker processes
        device: Device to run on (auto-detected if None)
        image_size: Target image size for ResNet-50
        
    Returns:
        FID score
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Calculating FID between two datasets...")
    print(f"Dataset 1: {len(dataset1)} samples")
    print(f"Dataset 2: {len(dataset2)} samples")
    
    # Load model once
    model = get_resnet50_features_model(device)
    
    # Extract features from both datasets
    print("\nExtracting features from dataset 1...")
    features1 = extract_features(
        dataset1,
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        image_size=image_size
    )
    
    print("\nExtracting features from dataset 2...")
    features2 = extract_features(
        dataset2,
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        image_size=image_size
    )
    
    # Calculate FID
    print("\nCalculating FID score...")
    fid_score = calculate_fid(features1, features2)
    
    return fid_score

