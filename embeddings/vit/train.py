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
        max_image_height=512,
        checkpoint_dir="experiments/checkpoints/mae",
    )
    # history: train_loss, val_loss, lr per epoch. With checkpoint_dir set,
    # ``mae_training_run.json`` is written there. For classification accuracy /
    # balanced accuracy, run ``train_linear_probe`` on the frozen encoder.

The dataset can be *any* ``torch.utils.data.Dataset`` whose ``__getitem__``
returns either:

* a dict with an ``'image'`` key containing a ``[1, H, W]`` grayscale tensor, or
* a plain ``[1, H, W]`` grayscale tensor.

Content is **never stretched** and **never letterboxed** to a fixed canvas.
Images whose height exceeds ``max_image_height`` are shrunk (aspect ratio
preserved) so that the new height equals ``max_image_height``; smaller images
are left untouched. Width is whatever aspect ratio produces — e.g. with
``max_image_height = 512``, an input of ``(H=1024, W=100)`` becomes
``(512, 50)`` and ``(H=500, W=500)`` stays ``(500, 500)``.

Because samples in a batch can now have different shapes, a custom collate
function (:func:`mae_pad_collate`) zero-pads each batch to the per-batch
maximum height / width, rounded up to the nearest multiple of ``patch_size``,
and emits a ``pad_mask`` marking the padded pixels so the model excludes
them from attention / loss.
"""

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from tqdm import tqdm

from embeddings.vit.model import MaskedAutoencoderViT, create_mae_vit
from embeddings.rank_me import compute_rank_me


def apply_max_height_shrink(
    img: torch.Tensor,
    max_image_height: Optional[int],
    interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    Only **shrinks** tall images. If the current height is at most
    ``max_image_height``, returns *img* unchanged. If it is **greater**, resizes
    so the new height is ``max_image_height`` and the width is scaled to keep
    aspect ratio. Never upscales.

    When *max_image_height* is ``None`` or not positive, returns *img* unchanged.
    """
    if max_image_height is None or max_image_height <= 0:
        return img
    _, h, w = img.shape
    if h <= max_image_height:
        return img
    new_w = max(1, int(round(w * max_image_height / h)))
    return T.functional.resize(
        img,
        [int(max_image_height), new_w],
        interpolation=interpolation,
        antialias=True,
    )


def standardize_per_image(
    img: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    """
    Per-image z-score normalisation: ``(img - mean) / std`` computed over all
    pixels of the sample.

    Ultrasound images have wildly varying dynamic range and contrast across
    machines / acquisitions. Standardising each image removes this low-level
    variation so the encoder can spend capacity on structure. After this
    transform the image has mean ≈ 0 and std ≈ 1, which also makes the zeros
    inserted by :func:`mae_pad_collate` sit at "content mean" rather than at
    "min pixel".
    """
    mean = img.mean()
    std = img.std().clamp_min(eps)
    return (img - mean) / std


def pad_to_patch_multiple(
    img: torch.Tensor, patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zero-pad ``[C, H, W]`` on the right / bottom so that both spatial sides are
    multiples of *patch_size*. Returns ``(padded, pad_mask)`` where ``pad_mask``
    is ``[1, H', W']`` boolean (True on padded pixels).
    """
    if img.dim() != 3:
        raise ValueError(f"pad_to_patch_multiple expects [C, H, W], got shape {tuple(img.shape)}")
    _, h, w = img.shape
    H = math.ceil(h / patch_size) * patch_size
    W = math.ceil(w / patch_size) * patch_size
    padded = T.functional.pad(img, [0, 0, W - w, H - h], fill=0)
    pad_mask = torch.ones(1, H, W, dtype=torch.bool)
    pad_mask[:, :h, :w] = False
    return padded, pad_mask


def resize_keep_aspect_pad(
    img: torch.Tensor,
    image_height: int,
    image_width: int,
    interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    :func:`apply_max_height_shrink` if needed, then letterbox into
    ``image_height × image_width`` (aspect ratio preserved, zero padding; no stretch).
    """
    padded, _ = resize_keep_aspect_pad_with_mask(
        img, image_height, image_width, interpolation=interpolation,
    )
    return padded


def resize_keep_aspect_pad_with_mask(
    img: torch.Tensor,
    image_height: int,
    image_width: int,
    interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    If the image is taller than *image_height*, shrinks to that height; then
    **letterboxes** into ``image_height × image_width`` (content scaled uniformly
    to **fit inside** the box, bars only — original aspect is preserved).

    Returns:
        padded_img: ``[C, image_height, image_width]``
        pad_mask:   ``[1, image_height, image_width]`` boolean, True where padding.
    """
    img = apply_max_height_shrink(img, image_height, interpolation=interpolation)
    th, tw = int(image_height), int(image_width)
    _, h, w = img.shape
    scale = min(th / h, tw / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = T.functional.resize(
        img, [new_h, new_w],
        interpolation=interpolation,
        antialias=True,
    )
    pad_h = th - new_h
    pad_w = tw - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = T.functional.pad(
        resized, [pad_left, pad_top, pad_right, pad_bottom], fill=0
    )

    pad_mask = torch.ones(1, th, tw, dtype=torch.bool)
    pad_mask[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = False
    return padded, pad_mask


# =====================================================================
# Augmentation pipeline
# =====================================================================

class MAEAugmentation(nn.Module):
    """
    Ultrasound-appropriate augmentations for MAE pre-training.

    Operates on a (height-capped) ``[1, H, W]`` tensor (no letterbox). Applies
    horizontal flip; optional rotation and Gaussian blur are wired up but
    currently disabled. No ``RandomResizedCrop`` — keep the content honest so
    the pad-mask produced by :func:`mae_pad_collate` stays meaningful.
    """

    def __init__(self) -> None:
        super().__init__()
        self.flip = T.RandomHorizontalFlip(p=0.5)
        self.rotation = T.RandomApply([T.RandomRotation(degrees=15)], p=0.3)
        self.blur = T.RandomApply(
            [T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2,
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 2:
            img = img.unsqueeze(0)
        img = self.flip(img)
        # img = self.rotation(img)
        # img = self.blur(img)
        return img


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
    2. :func:`apply_max_height_shrink` — shrink if height exceeds
       ``max_image_height`` (aspect kept); never upscale / stretch / pad.
    3. Apply augmentations (flip).
    4. :func:`standardize_per_image` — per-image z-score so the encoder sees
       contrast-normalised content (same normalisation is applied at
       validation / inference / probing time).

    Returns a single ``[1, H', W']`` tensor of **variable** shape. Per-batch
    padding is handled by :func:`mae_pad_collate`.
    """

    def __init__(self, max_image_height: int = 224, standardize: bool = True):
        self.max_image_height = int(max_image_height)
        self.standardize = bool(standardize)
        self.augmentation = MAEAugmentation()

    def __call__(self, sample: Any) -> torch.Tensor:
        image = _extract_image(sample)
        image = apply_max_height_shrink(image, self.max_image_height)
        image = self.augmentation(image)
        if self.standardize:
            image = standardize_per_image(image)
        return image


class MAEValTransform:
    """Validation transform: max-height shrink + per-image z-score, no augmentations."""

    def __init__(self, max_image_height: int = 224, standardize: bool = True):
        self.max_image_height = int(max_image_height)
        self.standardize = bool(standardize)

    def __call__(self, sample: Any) -> torch.Tensor:
        image = _extract_image(sample)
        image = apply_max_height_shrink(image, self.max_image_height)
        if self.standardize:
            image = standardize_per_image(image)
        return image


def mae_pad_collate(
    batch: List[torch.Tensor], patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate variable-shape ``[1, H_i, W_i]`` images into a single batch.

    The batch is zero-padded (right and bottom only) to ``(max_h, max_w)``,
    each rounded **up** to a multiple of ``patch_size``. Returns
    ``(images, pad_masks)``:

    * ``images``:    ``[B, C, H', W']`` float
    * ``pad_masks``: ``[B, 1, H', W']`` boolean, True where the pixel is padding.
    """
    if len(batch) == 0:
        raise ValueError("mae_pad_collate received an empty batch.")
    C = batch[0].shape[0]
    max_h = max(int(img.shape[-2]) for img in batch)
    max_w = max(int(img.shape[-1]) for img in batch)
    H = math.ceil(max_h / patch_size) * patch_size
    W = math.ceil(max_w / patch_size) * patch_size
    B = len(batch)
    images = torch.zeros(B, C, H, W, dtype=batch[0].dtype)
    pad_masks = torch.ones(B, 1, H, W, dtype=torch.bool)
    for i, img in enumerate(batch):
        _, h, w = img.shape
        images[i, :, :h, :w] = img
        pad_masks[i, :, :h, :w] = False
    return images, pad_masks


# =====================================================================
# Wrapped dataset (applies MAE transform on top of base dataset)
# =====================================================================

class _MAEDatasetWrapper(Dataset):
    """Wraps any dataset and applies the MAE transform.

    Each item is a ``[1, H, W]`` image tensor (variable shape). Padding into a
    fixed-shape batch is handled by :func:`mae_pad_collate`.
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
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, pad_masks in pbar:
        # non_blocking=True lets the CPU keep going while the transfer runs
        # (requires pin_memory=True on the DataLoader, which is already set).
        images    = images.to(device, non_blocking=True)
        pad_masks = pad_masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss, _, _ = model(images, mask_ratio=mask_ratio, pad_mask=pad_masks)

        # set_to_none=True frees gradient buffers instead of zeroing them — faster.
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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
) -> Tuple[float, float]:
    """Returns (val_loss, ref_mse).

    val_loss  — weighted training loss (l1/l2/fft weights as configured).
    ref_mse   — pure normalized per-patch MSE on masked patches, independent
                of loss weights; use this to compare runs with different configs.
    """
    model.eval()
    total_loss    = 0.0
    total_ref_mse = 0.0
    num_batches   = 0

    for images, pad_masks in dataloader:
        images    = images.to(device, non_blocking=True)
        pad_masks = pad_masks.to(device, non_blocking=True)
        loss, pred, mask = model(images, mask_ratio=mask_ratio, pad_mask=pad_masks)
        ref = model.reference_mse(images, pred, mask, pad_mask=pad_masks)
        total_loss    += loss.item()
        total_ref_mse += ref.item()
        num_batches   += 1

    n = max(num_batches, 1)
    return total_loss / n, total_ref_mse / n


def _save_train_transform_previews(
    train_dataset: Dataset,
    patch_size: int,
    out_dir: str,
    epoch_1based: int,
    num_samples: int,
) -> None:
    """Random training samples *after* MAE transform; also shows the pad-mask that
    :func:`mae_pad_collate` would produce when batching this single sample
    (padding to the next ``patch_size`` multiple). Saves a PNG in *out_dir*."""
    if num_samples <= 0:
        return
    n = len(train_dataset)
    if n == 0:
        return
    k = min(int(num_samples), n)
    indices = random.sample(range(n), k)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(k, 2, figsize=(7, 2.1 * k), squeeze=False)
    for row, idx in enumerate(indices):
        image = train_dataset[idx]
        padded, pad_mask = pad_to_patch_multiple(image, patch_size)
        im = padded.detach().float().cpu().numpy().squeeze(0)
        # Per-image min/max scaling so standardised (mean=0, std=1) previews
        # and raw [0, 1] previews both render with good contrast.
        im_lo, im_hi = float(im.min()), float(im.max())
        ax0 = axes[row, 0]
        ax0.imshow(im, cmap="gray", vmin=im_lo, vmax=im_hi)
        ax0.set_title(
            f"idx {idx} — image {tuple(image.shape[-2:])} → pad {tuple(padded.shape[-2:])}",
            fontsize=9,
        )
        ax0.axis("off")
        ax1 = axes[row, 1]
        ax1.imshow(
            pad_mask.detach().float().cpu().numpy().squeeze(0),
            cmap="cividis", vmin=0, vmax=1,
        )
        ax1.set_title("pad mask (1 = padded)", fontsize=9)
        ax1.axis("off")
    fig.suptitle(
        f"MAE training pipeline — after transforms (epoch {epoch_1based})",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = os.path.join(
        out_dir, f"mae_train_transform_preview_epoch{epoch_1based:04d}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → train transform preview: {out_path}")


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
        'max_image_height': model.max_image_height,
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
        'fft_loss_weight': model.fft_loss_weight,
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
    config.setdefault('fft_loss_weight', 0.0)
    # Legacy: older checkpoints stored a fixed letterbox canvas
    # (image_height, image_width or image_size). Re-interpret the height as
    # the new ``max_image_height`` cap and drop the width (no longer used).
    if 'max_image_height' not in config:
        if 'image_height' in config:
            config['max_image_height'] = int(config.pop('image_height'))
        elif 'image_size' in config:
            sz = config.pop('image_size')
            if isinstance(sz, int):
                config['max_image_height'] = int(sz)
            else:
                config['max_image_height'] = int(max(int(sz[0]), int(sz[1])))
    config.pop('image_width', None)
    config.pop('image_size', None)
    model = create_mae_vit(**config)
    bad = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if bad.missing_keys or bad.unexpected_keys:
        import warnings
        warnings.warn(
            f"Checkpoint loaded with strict=False: "
            f"{len(bad.missing_keys)} missing, {len(bad.unexpected_keys)} unexpected keys. "
            f"(Expected when moving from fixed-canvas PE to variable-shape RoPE.)",
            stacklevel=2,
        )
    model.eval()
    model.to(device)

    info = {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
    return model, info


def mae_model_summary(model: MaskedAutoencoderViT) -> Dict[str, Any]:
    """Serializable model architecture / hyperparameters + parameter counts."""
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'max_image_height': model.max_image_height,
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
        'fft_loss_weight': model.fft_loss_weight,
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

    val_tf = MAEValTransform(model.max_image_height or 0)

    if sample_indices is not None:
        indices = sample_indices[:num_samples]
    else:
        indices = torch.randperm(len(dataset))[:num_samples].tolist()

    num_show = len(indices)
    fig, axes = plt.subplots(num_show, 3, figsize=(12, 4 * num_show))
    if num_show == 1:
        axes = axes[None, :]

    for i, idx in enumerate(indices):
        img = val_tf(dataset[int(idx)])
        img, pad_mask_px = pad_to_patch_multiple(img, model.patch_size)
        img = img.unsqueeze(0).to(device)
        pad_mask_px = pad_mask_px.unsqueeze(0).to(device)

        with torch.no_grad():
            loss, pred, mask = model(
                img, mask_ratio=mask_ratio, pad_mask=pad_mask_px,
            )

        patches = model.patchify(img)
        mask_expanded = mask.unsqueeze(-1).expand_as(patches)

        grid_h = img.shape[-2] // model.patch_size
        grid_w = img.shape[-1] // model.patch_size

        masked_patches = patches * (1 - mask_expanded)
        masked_img = model.unpatchify(masked_patches, grid_h, grid_w)

        pred_pixels = pred
        if model.norm_pix_loss:
            mean = patches.mean(dim=-1, keepdim=True)
            var = patches.var(dim=-1, keepdim=True)
            pred_pixels = pred * (var + 1e-6).sqrt() + mean

        hybrid_patches = patches * (1 - mask_expanded) + pred_pixels * mask_expanded
        hybrid_img = model.unpatchify(hybrid_patches, grid_h, grid_w)

        def _to_np(t: torch.Tensor):
            return t.squeeze().detach().cpu().numpy()

        # Use the *original* image's pixel range to display all three panels so
        # they're on the same scale — works whether the transform produced [0, 1]
        # pixels or z-scored values.
        orig_np = _to_np(img)
        lo, hi = float(orig_np.min()), float(orig_np.max())

        axes[i, 0].imshow(orig_np, cmap='gray', vmin=lo, vmax=hi)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(_to_np(hybrid_img), cmap='gray', vmin=lo, vmax=hi)
        axes[i, 1].set_title(
            f'Reconstructed\n(loss={loss.item():.4f})'
        )
        axes[i, 1].axis('off')

        axes[i, 2].imshow(_to_np(masked_img), cmap='gray', vmin=lo, vmax=hi)
        axes[i, 2].set_title(f'Masked ({mask_ratio:.0%})')
        axes[i, 2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()


def dump_mae_training_metrics_artifacts(
    history: Dict[str, List[float]],
    out_dir: str,
    epoch_1based: int,
) -> Tuple[str, str]:
    """
    Persist MAE training curves up to the current epoch: a raw CSV and a PNG
    figure (loss linear, loss log, LR). Intended to be called on the same
    cadence as reconstruction dumps (e.g. every ``VISUALIZE_EVERY`` epochs in
    a notebook ``on_epoch_end`` callback).

    Filenames align with reconstruction naming::

        mae_metrics_epoch{epoch_1based:04d}.csv
        mae_metrics_epoch{epoch_1based:04d}.png

    Args:
        history: Dict with ``train_loss``, ``val_loss``, ``lr`` lists (same
                 semantics as :func:`train_mae` return value).
        out_dir:   Directory for both files (created if missing).
        epoch_1based: Tag in filenames; typically ``epoch + 1`` when invoked
                 from ``on_epoch_end(epoch, ...)``.

    Returns:
        ``(csv_path, png_path)`` absolute paths.
    """
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = str(Path(out_dir).resolve())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stem = f"mae_metrics_epoch{int(epoch_1based):04d}"
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    png_path = os.path.join(out_dir, f"{stem}.png")

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    lr_hist = history.get("lr", [])
    n = len(train_loss)
    if n == 0:
        return csv_path, png_path

    def _get(seq: List[float], i: int) -> float:
        return float(seq[i]) if i < len(seq) else float("nan")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "lr"])
        for i in range(n):
            w.writerow([i + 1, _get(train_loss, i), _get(val_loss, i), _get(lr_hist, i)])

    epochs_so_far = list(range(1, n + 1))
    tl = [float(train_loss[i]) for i in range(n)]
    vl = [_get(val_loss, i) for i in range(n)]
    lr_plot = [_get(lr_hist, i) for i in range(n)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    ax = axes[0]
    ax.plot(epochs_so_far, tl, label="train", linewidth=1.5)
    ax.plot(epochs_so_far, vl, label="val", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs_so_far, tl, label="train", linewidth=1.5)
    ax.plot(epochs_so_far, vl, label="val", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log)")
    ax.set_title("Loss (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs_so_far, lr_plot, color="tab:orange", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("LR schedule")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"MAE training metrics (through epoch {n})", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  → MAE metrics CSV: {csv_path}")
    print(f"  → MAE metrics plot: {png_path}")
    return csv_path, png_path


# =====================================================================
# Training plots
# =====================================================================

def save_mae_training_plots(history: Dict, out_dir: str) -> str:
    """
    Save a PNG with loss, reference MSE, and LR panels.
    Overwrites ``{out_dir}/training_plots.png`` each call.

    Returns the path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(history.get('train_loss', []))
    if n == 0:
        return ""
    xs = range(1, n + 1)
    ncols = 3

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    ax = axes[0]
    ax.plot(xs, history['train_loss'], label='train', linewidth=1.5)
    ax.plot(xs, history['val_loss'],   label='val',   linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Reconstruction Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(xs, history['val_ref_mse'], color='tab:green', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
    ax.set_title('Val Reference MSE')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(xs, history['lr'], color='tab:orange', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('LR')
    ax.set_title('LR Schedule')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"MAE training — epoch {n}", fontsize=11, y=1.01)
    fig.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, 'training_plots.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → training plots saved: {path}")
    return path


# =====================================================================
# Main entry point
# =====================================================================

def train_mae(
    dataset: Dataset,
    # --- model / transforms ---
    max_image_height: int = 224,
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
    fft_loss_weight: float = 0.0,
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
    rankme_every: int = 0,
    plot_every: int = 0,
    train_transform_preview: int = 4,
    # --- pretrained init ---
    pretrained_encoder: Optional[str] = None,
    # --- device / precision ---
    device: Optional[torch.device] = None,
    use_amp: bool = True,
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
        max_image_height: **Cap** on per-image height. Images taller are shrunk
                          to this (aspect preserved, no upscale); smaller images
                          pass through untouched. Must be divisible by
                          ``patch_size``. Content is **never** stretched and
                          **never** letterboxed to a fixed width.
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
        pretrained_encoder: timm model name to warm-start the encoder from, e.g.
                          ``"vit_small_patch16_224.dino"``. The RGB patch-embed
                          filters are averaged to produce a single grayscale
                          filter. The decoder stays randomly initialized.
                          Ignored when ``resume_from`` is set (checkpoint takes
                          precedence). ``None`` = random init (default).
        train_transform_preview: How many **random** training samples to plot (after
                          all transforms) each time a checkpoint is written; set to ``0`` to
                          disable. Only runs when *checkpoint_dir* is set. PNGs are saved
                          next to checkpoints as
                          ``mae_train_transform_preview_epoch{epoch}.png``.
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
    if max_image_height <= 0:
        raise ValueError("max_image_height must be positive.")
    if max_image_height % patch_size != 0:
        raise ValueError(
            f"max_image_height must be divisible by patch_size; "
            f"got max_image_height={max_image_height}, patch_size={patch_size}."
        )
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
    train_ds = _MAEDatasetWrapper(
        train_ds,
        MAETrainTransform(max_image_height=max_image_height),
    )
    val_ds = _MAEDatasetWrapper(
        val_ds, MAEValTransform(max_image_height=max_image_height),
    )

    def _collate(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return mae_pad_collate(batch, patch_size)

    _persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=_collate, persistent_workers=_persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=_collate, persistent_workers=_persistent,
    )

    # ---- model ----
    model = create_mae_vit(
        max_image_height=max_image_height,
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
        fft_loss_weight=fft_loss_weight,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(
        p.numel() for n, p in model.named_parameters()
        if not n.startswith('decoder') and not n.startswith('mask_token')
    )
    print(f"Model parameters: {n_params:,} total | {n_enc:,} encoder")

    if pretrained_encoder is not None and resume_from is None:
        from embeddings.pretrained_loader import load_pretrained_vit_encoder
        _pt_info = load_pretrained_vit_encoder(model, pretrained_encoder)
        print(f"Pretrained init: {_pt_info['summary']}")

    training_config: Dict[str, Any] = {
        'max_image_height': max_image_height,
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
        'fft_loss_weight': fft_loss_weight,
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
        'rankme_every': rankme_every,
        'plot_every': plot_every,
        'train_transform_preview': train_transform_preview,
        'pretrained_encoder': pretrained_encoder,
        'resume_from': resume_from,
        'n_train_samples': n_train,
        'n_val_samples': n_val,
        'device': str(device),
    }

    # ---- AMP scaler ----
    _use_amp = use_amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if _use_amp else None
    if _use_amp:
        print("Mixed precision (AMP) enabled.")

    # ---- optimizer & scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        min_lr=min_lr,
    )

    out_json = results_json
    if out_json is None and checkpoint_dir:
        out_json = os.path.join(checkpoint_dir, 'mae_training_run.json')

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

    # ---- history & metric definitions ----
    history: Dict[str, List] = {
        'train_loss': [], 'val_loss': [], 'val_ref_mse': [],
        'train_rank_me': [], 'val_rank_me': [],
        'lr': [],
    }

    metric_definitions = {
        'train_loss': 'Mean MAE reconstruction loss on training batches.',
        'val_loss': 'Mean MAE reconstruction loss on validation batches.',
        'val_ref_mse': 'Reference MSE (normalized per-patch) on masked validation patches.',
        'lr': 'Learning rate before optimizer step.',
    }

    # ---- restore history from existing JSON (enables correct resume) ----
    if start_epoch > 0 and out_json and os.path.isfile(out_json):
        try:
            with open(out_json, 'r', encoding='utf-8') as _f:
                _saved = json.load(_f).get('metrics_per_epoch', {})
            for key in history:
                if key in _saved and isinstance(_saved[key], list):
                    history[key] = list(_saved[key][:start_epoch])
            print(f"Restored {len(history['train_loss'])} epochs of history from {out_json}")
        except Exception as _e:
            print(f"Warning: could not load history from {out_json}: {_e}")

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
            model, train_loader, optimizer, device, mask_ratio, epoch, scaler,
        )
        val_loss, val_ref_mse = _validate(model, val_loader, device, mask_ratio)

        train_rank_me = val_rank_me = None

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_ref_mse'].append(val_ref_mse)
        history['train_rank_me'].append(train_rank_me)
        history['val_rank_me'].append(val_rank_me)
        history['lr'].append(current_lr)

        _rm = (
            f" | rank_me(tr/val)={train_rank_me:.1f}/{val_rank_me:.1f}"
            if val_rank_me is not None else ""
        )
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_ref_mse={val_ref_mse:.4f} | "
            f"lr={current_lr:.2e}"
            + _rm
        )

        if on_epoch_end is not None:
            on_epoch_end(epoch, history, model)

        # ---- checkpoint + optional transform preview (same cadence) ----
        if checkpoint_dir and (epoch + 1) % save_every == 0:
            path = os.path.join(checkpoint_dir, f"mae_epoch_{epoch+1}.pt")
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, train_loss, val_loss, path,
            )
            print(f"  → checkpoint saved: {path}")
            if out_json:
                dump_mae_training_results(
                    out_json, model=model, history=history,
                    training_config=training_config, metric_definitions=metric_definitions,
                )
            if train_transform_preview > 0:
                _save_train_transform_previews(
                    train_ds, patch_size, checkpoint_dir,
                    epoch + 1, train_transform_preview,
                )

        if checkpoint_dir and plot_every > 0 and (epoch + 1) % plot_every == 0:
            save_mae_training_plots(history, checkpoint_dir)

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

    if checkpoint_dir and plot_every > 0:
        save_mae_training_plots(history, checkpoint_dir)

    return model, history

