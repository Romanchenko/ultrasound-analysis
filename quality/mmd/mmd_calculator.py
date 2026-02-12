"""
MMD (Maximum Mean Discrepancy) calculator.

Supports multiple feature-extraction backbones via the shared
``quality.feature_extractor`` module:
    - 'resnet50'    : ImageNet pretrained ResNet-50
    - 'radimagenet' : RadImageNet pretrained ResNet-50 (from Keras H5)
    - nn.Module     : any custom feature extractor

MMD measures the distance between two probability distributions by
comparing their mean embeddings in a reproducing kernel Hilbert space
(RKHS).  A Gaussian (RBF) kernel is used by default.

    MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] − 2·E[k(x,y)]

Lower MMD means the two distributions are more similar; MMD = 0 implies
the distributions are identical (in the RKHS sense).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Optional, List, Union

from quality.fid.fid_calculator import extract_features
from quality.feature_extractor import get_features_model


# -----------------------------------------------------------------------
# Kernel helpers
# -----------------------------------------------------------------------

def _gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute the Gaussian (RBF) kernel matrix between rows of x and y.

        k(x, y) = exp(−||x − y||² / (2σ²))

    Args:
        x: Array of shape [n, d].
        y: Array of shape [m, d].
        sigma: Bandwidth parameter.

    Returns:
        Kernel matrix of shape [n, m].
    """
    # ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 * x_i . y_j
    x_sq = np.sum(x ** 2, axis=1, keepdims=True)  # [n, 1]
    y_sq = np.sum(y ** 2, axis=1, keepdims=True)  # [m, 1]
    dist_sq = x_sq + y_sq.T - 2.0 * (x @ y.T)    # [n, m]
    dist_sq = np.maximum(dist_sq, 0.0)             # numerical safety
    return np.exp(-dist_sq / (2.0 * sigma ** 2))


def _median_heuristic(features1: np.ndarray, features2: np.ndarray,
                      max_samples: int = 2000) -> float:
    """
    Estimate a good kernel bandwidth via the median heuristic:
    σ = median of pairwise distances between a subsample of both sets.

    Args:
        features1: [n1, d]
        features2: [n2, d]
        max_samples: Cap the subsample size to keep this fast.

    Returns:
        Bandwidth σ.
    """
    rng = np.random.RandomState(42)
    n1 = min(len(features1), max_samples)
    n2 = min(len(features2), max_samples)
    idx1 = rng.choice(len(features1), n1, replace=False)
    idx2 = rng.choice(len(features2), n2, replace=False)
    subset = np.concatenate([features1[idx1], features2[idx2]], axis=0)

    # Pairwise squared distances of the subsample
    sq = np.sum(subset ** 2, axis=1, keepdims=True)
    dist_sq = sq + sq.T - 2.0 * (subset @ subset.T)
    dist_sq = np.maximum(dist_sq, 0.0)

    # Take the median of the upper triangle (excluding diagonal)
    triu_idx = np.triu_indices_from(dist_sq, k=1)
    median_dist = float(np.median(np.sqrt(dist_sq[triu_idx])))

    # Avoid zero bandwidth
    return max(median_dist, 1e-6)


# -----------------------------------------------------------------------
# MMD computation
# -----------------------------------------------------------------------

def calculate_mmd(
    features1: np.ndarray,
    features2: np.ndarray,
    kernel: str = 'rbf',
    bandwidths: Optional[List[float]] = None,
    use_median_heuristic: bool = True,
) -> float:
    """
    Compute the (unbiased) Maximum Mean Discrepancy between two sets of
    feature vectors.

    When *bandwidths* is a list, a multi-scale kernel is used (the kernel
    values for each bandwidth are averaged).  When *bandwidths* is None
    and *use_median_heuristic* is True, the bandwidth is chosen
    automatically from the data via the median heuristic; additionally
    half and double that value are included for robustness.

    Args:
        features1: Features from first dataset  [n1, d].
        features2: Features from second dataset [n2, d].
        kernel:    Kernel type (currently only 'rbf' is supported).
        bandwidths: Explicit list of σ values.  If None, determined
                    automatically.
        use_median_heuristic: Whether to use the median heuristic when
                              *bandwidths* is None.

    Returns:
        MMD² value (float, ≥ 0; lower means more similar).
    """
    if kernel != 'rbf':
        raise ValueError(f"Unsupported kernel: {kernel}. Only 'rbf' is supported.")

    n = len(features1)
    m = len(features2)

    # --- decide bandwidths ---
    if bandwidths is None:
        if use_median_heuristic:
            sigma = _median_heuristic(features1, features2)
            bandwidths = [sigma * 0.5, sigma, sigma * 2.0]
        else:
            # Sensible fallback for 2048-dim ResNet features
            bandwidths = [1.0, 5.0, 10.0, 50.0, 100.0]

    # --- accumulate kernel values over all bandwidths ---
    k_xx_sum = np.zeros((n, n))
    k_yy_sum = np.zeros((m, m))
    k_xy_sum = np.zeros((n, m))

    for sigma in bandwidths:
        k_xx_sum += _gaussian_kernel(features1, features1, sigma)
        k_yy_sum += _gaussian_kernel(features2, features2, sigma)
        k_xy_sum += _gaussian_kernel(features1, features2, sigma)

    # Average over bandwidths
    num_bw = len(bandwidths)
    k_xx = k_xx_sum / num_bw
    k_yy = k_yy_sum / num_bw
    k_xy = k_xy_sum / num_bw

    # --- unbiased estimator of MMD² ---
    # Remove diagonal (i ≠ j only) for the self-kernel terms
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)

    mmd_sq = (
        k_xx.sum() / (n * (n - 1))
        + k_yy.sum() / (m * (m - 1))
        - 2.0 * k_xy.sum() / (n * m)
    )

    return float(mmd_sq)


# -----------------------------------------------------------------------
# High-level entry point
# -----------------------------------------------------------------------

def calculate_mmd_from_datasets(
    dataset1: Dataset,
    dataset2: Dataset,
    model_name: Union[str, nn.Module] = 'resnet50',
    weights_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    image_size: int = 224,
    bandwidths: Optional[List[float]] = None,
) -> float:
    """
    Calculate MMD between two PyTorch datasets.

    Args:
        dataset1:     First PyTorch Dataset.
        dataset2:     Second PyTorch Dataset.
        model_name:   Backbone selector (``'resnet50'``, ``'radimagenet'``,
                      or an ``nn.Module``).
        weights_path: Path to weight file (required for ``'radimagenet'``).
        batch_size:   Batch size for feature extraction.
        num_workers:  Number of DataLoader workers.
        device:       Device (auto-detected when *None*).
        image_size:   Input image size for the backbone.
        bandwidths:   Explicit kernel bandwidths.  If None, the median
                      heuristic is used.

    Returns:
        MMD² score (lower is better; 0 means identical distributions).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Calculating MMD between two datasets...")
    print(f"Dataset 1: {len(dataset1)} samples")
    print(f"Dataset 2: {len(dataset2)} samples")
    print(f"Feature extractor: {model_name if isinstance(model_name, str) else type(model_name).__name__}")

    # Load model once and share across both extractions
    model = get_features_model(
        model_name=model_name,
        device=device,
        weights_path=weights_path,
    )

    print("\nExtracting features from dataset 1...")
    features1 = extract_features(
        dataset1,
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        image_size=image_size,
    )

    print("\nExtracting features from dataset 2...")
    features2 = extract_features(
        dataset2,
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        image_size=image_size,
    )

    print("\nCalculating MMD score...")
    mmd_score = calculate_mmd(
        features1,
        features2,
        bandwidths=bandwidths,
    )

    return mmd_score
