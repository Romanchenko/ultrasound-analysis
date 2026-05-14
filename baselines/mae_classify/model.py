"""
LinearProbe model, variable-shape transforms, and collate for classification.
"""

import inspect
import math
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    """
    Frozen encoder + trainable linear classification head.

    Encoder can be any module with:
      - ``.encode(imgs[, pad_mask=...])`` → ``[B, embed_dim]``, or
      - ``.forward(imgs)`` → ``[B, embed_dim]``

    Features pass through a parameter-free BatchNorm before the linear head
    (He et al. 2022 §A.2): frozen MAE features have non-uniform per-dim scales,
    and BN typically lifts probe accuracy several points.
    """

    def __init__(self, encoder: nn.Module, embed_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.feat_bn = nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        self._use_encode = hasattr(encoder, "encode")
        self._encode_supports_pad_mask = False
        if self._use_encode:
            try:
                sig = inspect.signature(encoder.encode)
                self._encode_supports_pad_mask = "pad_mask" in sig.parameters
            except (TypeError, ValueError):
                pass

    def _get_embeddings(
        self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._use_encode:
            if self._encode_supports_pad_mask:
                return self.encoder.encode(x, pad_mask=pad_mask)
            return self.encoder.encode(x)
        return self.encoder(x)

    def forward(
        self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            emb = self._get_embeddings(x, pad_mask=pad_mask)
        emb = self.feat_bn(emb)
        return self.head(emb)


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def _apply_max_height_shrink(
    img: torch.Tensor, max_image_height: Optional[int],
) -> torch.Tensor:
    """Shrink so height ≤ max_image_height (aspect preserved, never upscale)."""
    if max_image_height is None or max_image_height <= 0:
        return img
    _, h, w = img.shape
    if h <= max_image_height:
        return img
    new_w = max(1, int(round(w * max_image_height / h)))
    return T.functional.resize(
        img,
        [int(max_image_height), new_w],
        interpolation=T.InterpolationMode.BILINEAR,
        antialias=True,
    )


def _standardize_per_image(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = img.mean()
    std = img.std().clamp_min(eps)
    return (img - mean) / std


def default_image_transform(
    max_image_height: int, standardize: bool = True,
) -> Callable[[dict], torch.Tensor]:
    """Build the default transform: grayscale → shrink → z-score per image."""

    def transform(sample: dict) -> torch.Tensor:
        img = sample["image"]
        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)
        if img.dim() == 2:
            img = img.unsqueeze(0)
        if img.size(0) > 1:
            img = img.mean(dim=0, keepdim=True)
        img = _apply_max_height_shrink(img, max_image_height)
        if standardize:
            img = _standardize_per_image(img)
        return img

    return transform


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def pad_classify_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]], patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate variable-shape ``[1, H_i, W_i]`` images + labels.

    Zero-pads (right/bottom) to ``(max_h, max_w)`` rounded up to a multiple of
    ``patch_size``. Returns ``(images, pad_masks, labels)``.
    """
    if not batch:
        raise ValueError("pad_classify_collate received an empty batch.")
    imgs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    C = imgs[0].shape[0]
    max_h = max(img.shape[-2] for img in imgs)
    max_w = max(img.shape[-1] for img in imgs)
    H = math.ceil(max_h / patch_size) * patch_size
    W = math.ceil(max_w / patch_size) * patch_size
    B = len(imgs)

    images = torch.zeros(B, C, H, W, dtype=imgs[0].dtype)
    pad_masks = torch.ones(B, 1, H, W, dtype=torch.bool)
    for i, img in enumerate(imgs):
        _, h, w = img.shape
        images[i, :, :h, :w] = img
        pad_masks[i, :, :h, :w] = False

    labels_t = torch.stack([
        lbl if isinstance(lbl, torch.Tensor) else torch.as_tensor(lbl)
        for lbl in labels
    ]).long()
    return images, pad_masks, labels_t


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationDatasetWrapper(Dataset):
    """Wraps any dataset returning sample dicts, applies image transform."""

    def __init__(self, base: Dataset, transform: Callable):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.base[idx]
        return self.transform(item), item["label_idx"]
