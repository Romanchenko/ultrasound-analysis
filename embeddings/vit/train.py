"""
MAE (Masked Autoencoder) training script for the custom ultrasound ViT.

Designed to be **Jupyter-notebook friendly**: call :func:`train_mae` from a
cell with a dataset and hyperparameters — no argparse required.

Usage (notebook)::

    from datasets_adapters.fetal_planes_db.fpd_dataset import FetalPlanesDBDataset
    from embeddings.vit.train import train_mae

    ds = FetalPlanesDBDataset(root="path/to/FETAL_PLANES_DB")
    model, history = train_mae(
        dataset=ds,
        epochs=100,
        batch_size=64,
        image_size=224,
        checkpoint_dir="experiments/checkpoints/mae",
    )
    # history: train_loss, val_loss, lr per epoch. With checkpoint_dir set,
    # ``mae_training_run.json`` is written there. For classification accuracy /
    # balanced accuracy, run ``train_linear_probe`` on the frozen encoder.

The dataset can be *any* ``torch.utils.data.Dataset`` whose ``__getitem__``
returns either:

* a dict with an ``'image'`` key containing a ``[1, H, W]`` grayscale tensor, or
* a plain ``[1, H, W]`` grayscale tensor.

Images are resized to ``image_size × image_size`` on the fly with
ultrasound-appropriate augmentations (random crop, flip, small rotation).
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from tqdm import tqdm

from embeddings.vit.model import MaskedAutoencoderViT, create_mae_vit


def resize_keep_aspect_pad(
    img: torch.Tensor,
    size: int,
    interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    Resize image to fit within size x size, keeping aspect ratio.
    Zero-pad on shorter sides to produce a square output.
    """
    padded, _ = resize_keep_aspect_pad_with_mask(img, size, interpolation)
    return padded


def resize_keep_aspect_pad_with_mask(
    img: torch.Tensor,
    size: int,
    interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resize image to fit within size x size, keeping aspect ratio.
    Zero-pad on shorter sides to produce a square output.

    Returns:
        padded_img: ``[C, size, size]``
        pad_mask:   ``[1, size, size]`` boolean, True where pixel is padding.
    """
    _, h, w = img.shape
    scale = min(size / h, size / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = T.functional.resize(
        img, [new_h, new_w],
        interpolation=interpolation,
        antialias=True,
    )
    pad_h = size - new_h
    pad_w = size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = T.functional.pad(
        resized, [pad_left, pad_top, pad_right, pad_bottom], fill=0
    )

    pad_mask = torch.ones(1, size, size, dtype=torch.bool)
    pad_mask[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = False
    return padded, pad_mask


# =====================================================================
# Augmentation pipeline
# =====================================================================

class MAEAugmentation(nn.Module):
    """
    Ultrasound-appropriate augmentations for MAE pre-training.

    Operates on an already-resized ``[1, image_size, image_size]`` tensor
    (after ``resize_keep_aspect_pad``).  Applies flip, small rotation,
    and Gaussian blur.  **No** ``RandomResizedCrop`` — that would
    destroy the known padding layout.

    When a ``pad_mask`` is given, all geometric transforms are applied to
    both the image and the mask so they stay aligned.
    """

    def __init__(self, image_size: int = 224):
        super().__init__()
        self.image_size = image_size
        self.flip = T.RandomHorizontalFlip(p=0.5)
        self.rotation = T.RandomApply([T.RandomRotation(degrees=15)], p=0.3)
        self.blur = T.RandomApply(
            [T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2,
        )

    def __call__(
        self,
        img: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if img.dim() == 2:
            img = img.unsqueeze(0)

        if pad_mask is not None:
            # Stack so geometric transforms apply identically to both
            combined = torch.cat([img, pad_mask.float()], dim=0)  # [2, H, W]
            combined = self.flip(combined)
            # combined = self.rotation(combined)
            img = combined[:1]
            pad_mask = combined[1:] > 0.5  # back to bool [1, H, W]

            # img = self.blur(img)
        else:
            img = self.flip(img)
            # img = self.rotation(img)
            # img = self.blur(img)

        return img, pad_mask


def _extract_image(sample: Any) -> torch.Tensor:
    """Pull a ``[1, H, W]`` grayscale tensor out of a dataset sample."""
    if isinstance(sample, dict):
        image = sample['image']
    elif isinstance(sample, (tuple, list)):
        image = sample[0]
    else:
        image = sample

    if not isinstance(image, torch.Tensor):
        image = T.ToTensor()(image)

    if image.dim() == 2:
        image = image.unsqueeze(0)
    if image.size(0) > 1:
        image = image.mean(dim=0, keepdim=True)

    return image


class MAETrainTransform:
    """
    Full transform applied to raw dataset samples before feeding to MAE.

    1. Extract / normalise to ``[1, H, W]``.
    2. ``resize_keep_aspect_pad_with_mask`` → ``(image, pad_mask)``.
    3. Apply augmentations (flip, rotation, blur) keeping pad_mask aligned.
    """

    def __init__(self, image_size: int = 224):
        self.augmentation = MAEAugmentation(image_size)
        self.image_size = image_size

    def __call__(self, sample: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        image = _extract_image(sample)
        image, pad_mask = resize_keep_aspect_pad_with_mask(
            image, self.image_size,
        )
        image, pad_mask = self.augmentation(image, pad_mask)
        return image, pad_mask


class MAEValTransform:
    """Validation transform: resize + pad, no augmentations. Returns ``(image, pad_mask)``."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def __call__(self, sample: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        image = _extract_image(sample)
        image, pad_mask = resize_keep_aspect_pad_with_mask(
            image, self.image_size,
        )
        return image, pad_mask


# =====================================================================
# Wrapped dataset (applies MAE transform on top of base dataset)
# =====================================================================

class _MAEDatasetWrapper(Dataset):
    """Wraps any dataset and applies the MAE transform.

    Each item is a ``(image, pad_mask)`` tuple.
    """

    def __init__(self, base_dataset: Dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.transform(self.base[idx])


# =====================================================================
# Cosine schedule with warmup
# =====================================================================

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine annealing to ``min_lr``."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            scale = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine
                for base_lr in self.base_lrs
            ]


# =====================================================================
# Single-epoch routines
# =====================================================================

def _train_one_epoch(
    model: MaskedAutoencoderViT,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    mask_ratio: float,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, pad_masks in pbar:
        images = images.to(device)
        pad_masks = pad_masks.to(device)
        loss, _, _ = model(images, mask_ratio=mask_ratio, pad_mask=pad_masks)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def _validate(
    model: MaskedAutoencoderViT,
    dataloader: DataLoader,
    device: torch.device,
    mask_ratio: float,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for images, pad_masks in dataloader:
        images = images.to(device)
        pad_masks = pad_masks.to(device)
        loss, _, _ = model(images, mask_ratio=mask_ratio, pad_mask=pad_masks)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# =====================================================================
# Checkpointing
# =====================================================================

def save_checkpoint(
    model: MaskedAutoencoderViT,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    path: str,
):
    """Save a training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Persist model config so we can re-create the architecture later
    model_config = {
        'image_size': model.image_size,
        'patch_size': model.patch_size,
        'in_channels': model.in_channels,
        'embed_dim': model.embed_dim,
        'depth': model.depth,
        'num_heads': model.num_heads,
        'decoder_embed_dim': model.decoder_embed_dim,
        'decoder_depth': model.decoder_depth,
        'decoder_num_heads': model.decoder_num_heads,
        'mlp_ratio': model.mlp_ratio,
        'norm_pix_loss': model.norm_pix_loss,
        'clip_pixel_pred': model.clip_pixel_pred,
        'decoder_pred_num_layers': model.decoder_pred_num_layers,
        'decoder_pred_hidden_dim': model.decoder_pred_hidden_dim,
        'l1_loss_weight': model.l1_loss_weight,
        'l2_loss_weight': model.l2_loss_weight,
    }

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_config': model_config,
    }, path)


def load_checkpoint(
    path: str,
    device: Optional[torch.device] = None,
) -> Tuple[MaskedAutoencoderViT, Dict[str, Any]]:
    """
    Load a MAE ViT from a training checkpoint.

    Returns:
        model:  Loaded ``MaskedAutoencoderViT`` in eval mode.
        info:   Dict with ``epoch``, ``train_loss``, ``val_loss``, etc.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(path, map_location='cpu')

    config = dict(ckpt['model_config'])
    # Legacy keys from older checkpoints
    config.pop('pixel_min', None)
    config.pop('pixel_max', None)
    config.setdefault('l1_loss_weight', 0.0)
    config.setdefault('l2_loss_weight', 1.0)
    model = create_mae_vit(**config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(device)

    info = {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
    return model, info


def mae_model_summary(model: MaskedAutoencoderViT) -> Dict[str, Any]:
    """Serializable model architecture / hyperparameters + parameter counts."""
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'image_size': model.image_size,
        'patch_size': model.patch_size,
        'in_channels': model.in_channels,
        'embed_dim': model.embed_dim,
        'depth': model.depth,
        'num_heads': model.num_heads,
        'decoder_embed_dim': model.decoder_embed_dim,
        'decoder_depth': model.decoder_depth,
        'decoder_num_heads': model.decoder_num_heads,
        'mlp_ratio': model.mlp_ratio,
        'norm_pix_loss': model.norm_pix_loss,
        'clip_pixel_pred': model.clip_pixel_pred,
        'decoder_pred_num_layers': model.decoder_pred_num_layers,
        'decoder_pred_hidden_dim': model.decoder_pred_hidden_dim,
        'l1_loss_weight': model.l1_loss_weight,
        'l2_loss_weight': model.l2_loss_weight,
        'num_parameters': n_params,
        'num_trainable_parameters': n_trainable,
    }


def dump_mae_training_results(
    path: str,
    *,
    model: MaskedAutoencoderViT,
    history: Dict[str, List[float]],
    training_config: Dict[str, Any],
    metric_definitions: Optional[Dict[str, str]] = None,
) -> str:
    """
    Write one JSON file with model config, training hyperparameters, and per-epoch loss curves.

    Classification **accuracy** / **balanced accuracy** are not part of MAE pre-training;
    evaluate the frozen encoder on a labeled task (e.g. ``train_linear_probe``).

    Returns:
        Absolute path written (``path`` resolved).
    """
    path = str(Path(path).resolve())
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        'model': mae_model_summary(model),
        'training': training_config,
        'metrics_per_epoch': dict(history),
        'downstream_metrics': (
            'Per-epoch accuracy and balanced accuracy on a supervised task are obtained by '
            'linear probing the frozen encoder, e.g. '
            'datasets_adapters.fetal_planes_db.linear_probe.train_linear_probe.'
        ),
    }
    if metric_definitions is not None:
        payload['metric_definitions'] = metric_definitions

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(f"Training run log written to {path}")
    return path


# =====================================================================
# Visualisation helpers
# =====================================================================

def visualize_reconstruction(
    model: MaskedAutoencoderViT,
    dataset: Dataset,
    num_samples: int = 4,
    mask_ratio: float = 0.75,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    sample_indices: Optional[List[int]] = None,
):
    """
    Show three views per sample:

    1. **Original** -- full image.
    2. **Reconstructed** -- original pixels on kept patches, model
       reconstruction on masked patches.
    3. **Masked** -- visible patches only; masked patches set to zero.

    Args:
        sample_indices: Fixed dataset indices to use. If ``None``, random
            indices are chosen. Providing fixed indices makes the
            visualisation reproducible across epochs / runs.

    Works in both Jupyter notebooks and script environments.
    """
    import matplotlib.pyplot as plt

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    val_tf = MAEValTransform(model.image_size)

    if sample_indices is not None:
        indices = sample_indices[:num_samples]
    else:
        indices = torch.randperm(len(dataset))[:num_samples].tolist()

    num_show = len(indices)
    fig, axes = plt.subplots(num_show, 3, figsize=(12, 4 * num_show))
    if num_show == 1:
        axes = axes[None, :]

    for i, idx in enumerate(indices):
        img, pad_mask_px = val_tf(dataset[int(idx)])
        img = img.unsqueeze(0).to(device)
        pad_mask_px = pad_mask_px.unsqueeze(0).to(device)

        with torch.no_grad():
            loss, pred, mask = model(
                img, mask_ratio=mask_ratio, pad_mask=pad_mask_px,
            )

        patches = model.patchify(img)
        mask_expanded = mask.unsqueeze(-1).expand_as(patches)

        masked_patches = patches * (1 - mask_expanded)
        masked_img = model.unpatchify(masked_patches)

        pred_pixels = pred
        if model.norm_pix_loss:
            mean = patches.mean(dim=-1, keepdim=True)
            var = patches.var(dim=-1, keepdim=True)
            pred_pixels = pred * (var + 1e-6).sqrt() + mean

        hybrid_patches = patches * (1 - mask_expanded) + pred_pixels * mask_expanded
        hybrid_img = model.unpatchify(hybrid_patches)

        def _to_np(t):
            return t.squeeze().cpu().numpy().clip(0, 1)

        axes[i, 0].imshow(_to_np(img), cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(_to_np(hybrid_img), cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(
            f'Reconstructed\n(loss={loss.item():.4f})'
        )
        axes[i, 1].axis('off')

        axes[i, 2].imshow(_to_np(masked_img), cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Masked ({mask_ratio:.0%})')
        axes[i, 2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()


# =====================================================================
# Main entry point
# =====================================================================

def train_mae(
    dataset: Dataset,
    # --- model ---
    image_size: int = 224,
    patch_size: int = 16,
    embed_dim: int = 384,
    depth: int = 6,
    num_heads: int = 6,
    decoder_embed_dim: int = 192,
    decoder_depth: int = 4,
    decoder_num_heads: int = 3,
    mlp_ratio: float = 4.0,
    norm_pix_loss: bool = True,
    clip_pixel_pred: bool = True,
    decoder_pred_num_layers: int = 1,
    decoder_pred_hidden_dim: Optional[int] = None,
    l1_loss_weight: float = 0.0,
    l2_loss_weight: float = 1.0,
    # --- training ---
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1.5e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 10,
    min_lr: float = 1e-6,
    mask_ratio: float = 0.75,
    val_split: float = 0.1,
    num_workers: int = 4,
    # --- checkpointing ---
    checkpoint_dir: Optional[str] = None,
    save_every: int = 10,
    # --- device ---
    device: Optional[torch.device] = None,
    # --- resume ---
    resume_from: Optional[str] = None,
    # --- logging ---
    results_json: Optional[str] = None,
    on_epoch_end: Optional[
        Callable[[int, Dict[str, List[float]], MaskedAutoencoderViT], None]
    ] = None,
) -> Tuple[MaskedAutoencoderViT, Dict[str, List[float]]]:
    """
    Train a MAE ViT from scratch on a grayscale ultrasound dataset.

    Args:
        dataset:          Any ``torch.utils.data.Dataset`` returning
                          grayscale image tensors (or dicts with ``'image'``).
        image_size:       Input image size (square).
        patch_size:       Patch side length.
        embed_dim:        Encoder embedding dimension.
        depth:            Encoder depth.
        num_heads:        Encoder attention heads.
        decoder_embed_dim: Decoder embedding dimension.
        decoder_depth:    Decoder depth.
        decoder_num_heads: Decoder attention heads.
        mlp_ratio:        MLP expansion ratio.
        norm_pix_loss:    Per-patch normalisation in loss.
        clip_pixel_pred:  If True, apply sigmoid to decoder pixel predictions.
                          Should be False when ``norm_pix_loss=True`` (normalised
                          targets extend outside [0, 1], which sigmoid cannot reach).
        epochs:           Total training epochs.
        batch_size:       Batch size.
        lr:               Peak learning rate.
        weight_decay:     AdamW weight decay.
        warmup_epochs:    Linear warmup epochs.
        min_lr:           Minimum LR after cosine annealing.
        mask_ratio:       Fraction of patches to mask.
        val_split:        Fraction of data held out for validation.
        num_workers:      DataLoader workers.
        checkpoint_dir:   Where to save checkpoints.  ``None`` = no saving.
        save_every:       Save a checkpoint every N epochs.
        device:           Training device (auto-detected when None).
        resume_from:      Path to a checkpoint to resume training from.
        results_json:     If set, write run summary (model, training, loss/lr per epoch) to this path.
                          If ``None`` but ``checkpoint_dir`` is set, writes ``mae_training_run.json``
                          there.
        decoder_pred_num_layers: Passed to :func:`create_mae_vit`.
        decoder_pred_hidden_dim: Passed to :func:`create_mae_vit`.
        l1_loss_weight:   Weight on L1 (mean abs error per patch) in reconstruction loss.
        l2_loss_weight:   Weight on L2 (mean squared error per patch). Default pure L2.
        on_epoch_end:     Optional callback ``(epoch, history, model)`` invoked after each epoch
                          (``epoch`` is 0-based). Useful for live plots in notebooks.

    Returns:
        model:   Trained ``MaskedAutoencoderViT``.
        history: Dict with per-epoch ``train_loss``, ``val_loss``, ``lr``.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # ---- split dataset ----
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {n_train} samples | Val: {n_val} samples")

    # ---- wrap with transforms ----
    train_ds = _MAEDatasetWrapper(train_ds, MAETrainTransform(image_size))
    val_ds = _MAEDatasetWrapper(val_ds, MAEValTransform(image_size))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ---- model ----
    model = create_mae_vit(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=1,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=mlp_ratio,
        norm_pix_loss=norm_pix_loss,
        clip_pixel_pred=clip_pixel_pred,
        decoder_pred_num_layers=decoder_pred_num_layers,
        decoder_pred_hidden_dim=decoder_pred_hidden_dim,
        l1_loss_weight=l1_loss_weight,
        l2_loss_weight=l2_loss_weight,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(
        p.numel() for n, p in model.named_parameters()
        if not n.startswith('decoder') and not n.startswith('mask_token')
    )
    print(f"Model parameters: {n_params:,} total | {n_enc:,} encoder")

    training_config: Dict[str, Any] = {
        'image_size': image_size,
        'patch_size': patch_size,
        'embed_dim': embed_dim,
        'depth': depth,
        'num_heads': num_heads,
        'decoder_embed_dim': decoder_embed_dim,
        'decoder_depth': decoder_depth,
        'decoder_num_heads': decoder_num_heads,
        'mlp_ratio': mlp_ratio,
        'norm_pix_loss': norm_pix_loss,
        'decoder_pred_num_layers': decoder_pred_num_layers,
        'decoder_pred_hidden_dim': decoder_pred_hidden_dim,
        'l1_loss_weight': l1_loss_weight,
        'l2_loss_weight': l2_loss_weight,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'warmup_epochs': warmup_epochs,
        'min_lr': min_lr,
        'mask_ratio': mask_ratio,
        'val_split': val_split,
        'num_workers': num_workers,
        'checkpoint_dir': checkpoint_dir,
        'save_every': save_every,
        'resume_from': resume_from,
        'n_train_samples': n_train,
        'n_val_samples': n_val,
        'device': str(device),
    }

    # ---- optimizer & scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        min_lr=min_lr,
    )

    start_epoch = 0

    # ---- resume ----
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # ---- training loop ----
    history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': [], 'lr': []}

    metric_definitions = {
        'train_loss': 'Mean MAE reconstruction loss on training batches.',
        'val_loss': 'Mean MAE reconstruction loss on validation batches.',
        'lr': 'Learning rate before optimizer step.',
    }

    out_json = results_json
    if out_json is None and checkpoint_dir:
        out_json = os.path.join(checkpoint_dir, 'mae_training_run.json')

    if out_json:
        dump_mae_training_results(
            out_json,
            model=model,
            history=history,
            training_config=training_config,
            metric_definitions=metric_definitions,
        )

    for epoch in range(start_epoch, epochs):
        current_lr = optimizer.param_groups[0]['lr']
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, device, mask_ratio, epoch,
        )
        val_loss = _validate(model, val_loader, device, mask_ratio)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if on_epoch_end is not None:
            on_epoch_end(epoch, history, model)

        # ---- checkpoint ----
        if checkpoint_dir and (epoch + 1) % save_every == 0:
            path = os.path.join(checkpoint_dir, f"mae_epoch_{epoch+1}.pt")
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, train_loss, val_loss, path,
            )
            print(f"  → checkpoint saved: {path}")

    # ---- final checkpoint ----
    if checkpoint_dir:
        path = os.path.join(checkpoint_dir, "mae_final.pt")
        save_checkpoint(
            model, optimizer, scheduler,
            epochs - 1, history['train_loss'][-1],
            history['val_loss'][-1] if history['val_loss'] else None,
            path,
        )
        print(f"Final checkpoint saved: {path}")

    
    if out_json:
        dump_mae_training_results(
            out_json,
            model=model,
            history=history,
            training_config=training_config,
            metric_definitions=metric_definitions,
        )

    return model, history

