"""
Dataset class for Fetal Head Circumference dataset.
Loads grayscale ultrasound images with configurable transformations.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Optional, Callable
from pathlib import Path
import torchvision.transforms as transforms

# Set matplotlib backend for notebooks (optional, but can help with display)
try:
    get_ipython()  # Check if running in IPython/Jupyter
    import matplotlib
    matplotlib.use('TkAgg')  # Use interactive backend in notebooks
except NameError:
    pass  # Not in IPython, use default backend
except ImportError:
    pass  # matplotlib not available


class FetalHeadCircDataset(Dataset):
    """
    Dataset class for Fetal Head Circumference grayscale ultrasound images.
    
    Dataset structure:
        images_dir/
            <filename>.png
            <filename>_Annotation.png (for training set only)
        csv_file contains: filename, pixel size, head circumference (mm)
    
    Args:
        images_dir: Directory containing images (train or test folder)
        csv_file: Path to CSV file with metadata
        transform: List of transformation functions to apply (default: resize_with_pad)
        target_size: Target size for resize_with_pad (default: (224, 224))
        load_annotations: Whether to load annotation images (default: True for training)
    """
    
    def __init__(
        self,
        images_dir: str,
        csv_file: str,
        transform: Optional[List[Callable]] = None,
        target_size: tuple = (224, 224),
        load_annotations: bool = True
    ):
        self.images_dir = Path(images_dir)
        self.csv_file = Path(csv_file)
        self.target_size = target_size
        self.load_annotations = load_annotations
        
        # Validate paths
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory does not exist: {self.images_dir}")
        
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file does not exist: {self.csv_file}")
        
        # Load CSV file
        self.df = pd.read_csv(self.csv_file)
        
        # Verify required columns exist
        required_columns = ['filename', 'pixel size', 'head circumference (mm)']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"CSV file missing required columns: {missing_columns}")
        
        # Build image paths and metadata
        self.image_paths = []
        self.annotation_paths = []
        self.metadata = []
        
        for idx, row in self.df.iterrows():
            filename = row['filename']
            # Remove .png extension if present (CSV might have it)
            image_name = filename.replace('.png', '')
            image_path = self.images_dir / f"{image_name}.png"
            
            if image_path.exists():
                self.image_paths.append(image_path)
                
                # Try to load annotation if requested
                annotation_path = None
                if self.load_annotations:
                    annotation_path = self.images_dir / f"{image_name}_Annotation.png"
                    if annotation_path.exists():
                        self.annotation_paths.append(annotation_path)
                    else:
                        self.annotation_paths.append(None)
                        print(f"Warning: Annotation not found for {image_name}")
                else:
                    self.annotation_paths.append(None)
                
                # Store metadata
                self.metadata.append({
                    'filename': filename,
                    'pixel_size': row.get('pixel size', None),
                    'head_circumference': row.get('head circumference (mm)', None),
                    'image_name': image_name
                })
            else:
                print(f"Warning: Image not found: {image_path}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in {self.images_dir}")
        
        print(f"Loaded {len(self.image_paths)} images from {self.images_dir}")
        if self.load_annotations:
            annotation_count = sum(1 for p in self.annotation_paths if p is not None)
            print(f"Found {annotation_count} annotation images")
        
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
        try:
            from torchtune.modules.transforms.vision_utils.resize_with_pad import resize_with_pad
        except ImportError:
            raise ImportError(
                "torchtune is required for resize_with_pad. "
                "Install it with: pip install torchtune"
            )
        
        # Create resize_with_pad transform with zero padding
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
                - 'annotation': Annotation tensor [1, H, W] or None (if load_annotations=True and annotation exists)
                - 'metadata': Dictionary with metadata (filename, pixel_size, head_circumference, image_name)
        """
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]
        metadata = self.metadata[idx]
        
        # Load image as grayscale
        try:
            image = Image.open(image_path).convert('L')  # 'L' mode for grayscale
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('L', self.target_size, color=0)
        
        # Load annotation if available
        annotation = None
        if annotation_path is not None and annotation_path.exists():
            try:
                annotation = Image.open(annotation_path).convert('L')  # Grayscale annotation
            except Exception as e:
                print(f"Error loading annotation {annotation_path}: {e}")
        
        # Apply transforms to image
        if self.transform:
            image = self.transform(image)
        else:
            # Default: just convert to tensor
            image = transforms.ToTensor()(image)
        
        # Apply same transforms to annotation if it exists
        if annotation is not None:
            if self.transform:
                # For annotation, we might want to use nearest neighbor for resizing
                # But for simplicity, use the same transform
                annotation = self.transform(annotation)
            else:
                annotation = transforms.ToTensor()(annotation)
        
        # Ensure image is single channel tensor [1, H, W]
        if image.dim() == 2:
            image = image.unsqueeze(0)  # Add channel dimension
        elif image.dim() == 3 and image.size(0) > 1:
            # If multiple channels, take first channel (shouldn't happen for grayscale)
            image = image[0:1]
        
        # Ensure annotation is single channel tensor [1, H, W] if it exists
        if annotation is not None:
            if annotation.dim() == 2:
                annotation = annotation.unsqueeze(0)
            elif annotation.dim() == 3 and annotation.size(0) > 1:
                annotation = annotation[0:1]
        
        result = {
            'image': image,
            'metadata': metadata
        }
        
        if annotation is not None:
            result['annotation'] = annotation
        
        return result
    
    def get_statistics(self) -> dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics about pixel size and head circumference
        """
        pixel_sizes = [m['pixel_size'] for m in self.metadata if m['pixel_size'] is not None]
        head_circs = [m['head_circumference'] for m in self.metadata if m['head_circumference'] is not None]
        
        stats = {
            'num_images': len(self.image_paths),
            'num_annotations': sum(1 for p in self.annotation_paths if p is not None)
        }
        
        if pixel_sizes:
            stats['pixel_size'] = {
                'mean': float(pd.Series(pixel_sizes).mean()),
                'std': float(pd.Series(pixel_sizes).std()),
                'min': float(pd.Series(pixel_sizes).min()),
                'max': float(pd.Series(pixel_sizes).max())
            }
        
        if head_circs:
            stats['head_circumference'] = {
                'mean': float(pd.Series(head_circs).mean()),
                'std': float(pd.Series(head_circs).std()),
                'min': float(pd.Series(head_circs).min()),
                'max': float(pd.Series(head_circs).max())
            }
        
        return stats


def visualize_transformation(
    dataset: FetalHeadCircDataset,
    image_name: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 7)
):
    """
    Visualize transformation effects on an image from the dataset.
    Shows before and after images side by side.
    
    Args:
        dataset: FetalHeadCircDataset instance
        image_name: Name of the image to visualize (without .png extension).
                    If None, a random image is selected.
        save_path: Optional path to save the figure. If None, displays the figure.
        figsize: Figure size (width, height) for matplotlib
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    
    # Select image
    if image_name is None:
        # Select random image
        idx = random.randint(0, len(dataset) - 1)
        image_name = dataset.metadata[idx]['image_name']
        print(f"Selected random image: {image_name}")
    else:
        # Find image by name
        idx = None
        for i, metadata in enumerate(dataset.metadata):
            if metadata['image_name'] == image_name or metadata['filename'] == image_name:
                idx = i
                break
        
        if idx is None:
            raise ValueError(f"Image '{image_name}' not found in dataset")
    
    # Get the transformed sample from dataset
    sample = dataset[idx]
    transformed_image = sample['image']
    
    # Load original image without transforms
    image_path = dataset.image_paths[idx]
    original_image = Image.open(image_path).convert('L')
    
    # Convert to numpy for visualization
    def tensor_to_numpy(img_tensor):
        """Convert image tensor to numpy array for visualization."""
        if isinstance(img_tensor, torch.Tensor):
            # Remove batch dimension if present
            if img_tensor.dim() == 4:
                img_tensor = img_tensor[0]
            # [C, H, W] or [H, W] -> [H, W]
            if img_tensor.dim() == 3:
                img_tensor = img_tensor[0]  # Take first channel for grayscale
            img_np = img_tensor.detach().cpu().numpy()
            # Clip to [0, 1] range
            img_np = np.clip(img_np, 0, 1)
            return img_np
        else:
            # PIL Image
            return np.array(img_tensor) / 255.0
    
    original_np = np.array(original_image) / 255.0
    transformed_np = tensor_to_numpy(transformed_image)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Original image
    axes[0].imshow(original_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Original: {image_name}\nSize: {original_image.size}', fontsize=12)
    axes[0].axis('off')
    
    # Transformed image
    axes[1].imshow(transformed_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Transformed: {image_name}\nSize: {transformed_np.shape}', fontsize=12)
    axes[1].axis('off')
    
    # Add metadata if available
    metadata = dataset.metadata[idx]
    metadata_text = f"Pixel size: {metadata.get('pixel_size', 'N/A')}\n"
    metadata_text += f"Head circumference: {metadata.get('head_circumference', 'N/A')} mm"
    
    fig.suptitle(metadata_text, fontsize=10, y=0.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()
        # Don't close in notebook mode
        # plt.close()


def visualize_multiple_transformations(
    dataset: FetalHeadCircDataset,
    num_samples: int = 4,
    save_path: Optional[str] = None,
    figsize: tuple = (20, 10)
):
    """
    Visualize transformations on multiple random images.
    
    Args:
        dataset: FetalHeadCircDataset instance
        num_samples: Number of random images to visualize
        save_path: Optional path to save the figure
        figsize: Figure size (width, height) for matplotlib
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # Get transformed sample
        sample = dataset[idx]
        transformed_image = sample['image']
        
        # Load original
        image_path = dataset.image_paths[idx]
        original_image = Image.open(image_path).convert('L')
        
        # Convert to numpy
        original_np = np.array(original_image) / 255.0
        
        # Convert tensor to numpy
        if isinstance(transformed_image, torch.Tensor):
            if transformed_image.dim() == 3:
                transformed_np = transformed_image[0].detach().cpu().numpy()
            else:
                transformed_np = transformed_image.detach().cpu().numpy()
            transformed_np = np.clip(transformed_np, 0, 1)
        else:
            transformed_np = np.array(transformed_image) / 255.0
        
        # Original
        axes[i, 0].imshow(original_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Original: {dataset.metadata[idx]["image_name"]}', fontsize=10)
        axes[i, 0].axis('off')
        
        # Transformed
        axes[i, 1].imshow(transformed_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Transformed: {dataset.metadata[idx]["image_name"]}', fontsize=10)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()
        # Don't close in notebook mode
        # plt.close()

