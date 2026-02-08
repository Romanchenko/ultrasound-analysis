# Frechet Inception Distance (FID) Calculator

This module calculates FID scores between two datasets to measure their similarity.

## Overview

FID measures the distance between two datasets' feature distributions. Lower FID scores indicate more similar datasets.

## Features

- Uses pretrained ResNet-50 for feature extraction
- Handles grayscale images (converts to RGB automatically)
- Handles images of different sizes (resizes to 224x224)
- Works with any PyTorch Dataset

## Usage

### As a Python Module

```python
from quality.fid import calculate_fid_from_datasets
from datasets_adapters.fetal_planes_db.fpd_dataset import FetalPlanesDBDataset

# Load datasets
dataset1 = FetalPlanesDBDataset(root='path/to/dataset1')
dataset2 = FetalPlanesDBDataset(root='path/to/dataset2')

# Calculate FID
fid_score = calculate_fid_from_datasets(dataset1, dataset2)
print(f"FID Score: {fid_score:.4f}")
```

## API Reference

### `calculate_fid_from_datasets(dataset1, dataset2, ...)`

Calculate FID between two PyTorch datasets.

**Parameters:**
- `dataset1`: First PyTorch Dataset
- `dataset2`: Second PyTorch Dataset
- `batch_size`: Batch size (default: 32)
- `num_workers`: Number of workers (default: 4)
- `device`: Device to use (default: auto-detect)
- `image_size`: Target image size (default: 224)

**Returns:**
- FID score (float, lower is better)

### `extract_features(dataset, ...)`

Extract features from a dataset using ResNet-50.

**Parameters:**
- `dataset`: PyTorch Dataset
- `model`: Optional pretrained model (if None, ResNet-50 is loaded)
- `batch_size`: Batch size (default: 32)
- `num_workers`: Number of workers (default: 4)
- `device`: Device to use (default: auto-detect)
- `image_size`: Target image size (default: 224)

**Returns:**
- Numpy array of features [num_samples, feature_dim]

### `calculate_fid(features1, features2)`

Calculate FID between two feature arrays.

**Parameters:**
- `features1`: Features from first dataset [num_samples1, feature_dim]
- `features2`: Features from second dataset [num_samples2, feature_dim]
- `eps`: Small value for numerical stability (default: 1e-6)

**Returns:**
- FID score (float)

## How It Works

1. **Feature Extraction**: Uses pretrained ResNet-50 (ImageNet weights) to extract features from images
2. **Image Preprocessing**: 
   - Converts grayscale to RGB (replicates channel)
   - Resizes to 224x224 (ResNet-50 input size)
   - Normalizes using ImageNet statistics
3. **FID Calculation**:
   - Computes mean and covariance of features for each dataset
   - Calculates: `FID = ||μ₁ - μ₂||² + Tr(C₁ + C₂ - 2√(C₁C₂))`
   - Lower scores indicate more similar distributions

## Notes

- FID score of 0 means identical distributions
- Lower FID scores indicate more similar datasets
- The score is sensitive to the number of samples - more samples give more reliable estimates
- ResNet-50 features are extracted from the layer before the classification head (2048-dimensional features)
