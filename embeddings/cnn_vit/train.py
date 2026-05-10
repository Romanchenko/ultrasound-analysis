"""
MAE training script for the CNN + ViT ultrasound model.

Identical workflow to ``embeddings/vit/train.py`` with the following additions:

* The collate function pads images to multiples of
  ``pixel_patch_size = patch_size * cnn_effective_stride`` (instead of
  ``patch_size``) so that after CNN downsampling the spatial grid is still
  divisible by ``patch_size``.
* ``train_mae`` accepts three extra CNN configuration arguments:
  ``cnn_layer_channels``, ``cnn_kernel_sizes``, ``cnn_strides``.
* Checkpoints include CNN config; ``load_checkpoint`` reconstructs the full
  CNN + ViT model.
* Visualisation uses ``model.pixel_patch_size`` for patchify / grid arithmetic.

Usage (notebook)::

    from embeddings.cnn_vit.train import train_mae

    model, history = train_mae(
        dataset=ds,
        cnn_layer_channels=[32, 64],
        cnn_kernel_sizes=3,
        cnn_strides=[1, 2],   # effective stride 2 → pixel_patch_size = patch_size * 2
        max_image_height=224,
        patch_size=16,
        checkpoint_dir="experiments/checkpoints/cnn_mae",
    )
"""

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from tqdm import tqdm

from embeddings.cnn_vit.model import CNNMaskedAutoencoderViT, create_cnn_mae_vit
from embeddings.rank_me import compute_rank_me, collect_embeddings


def apply_max_height_shrink(
    img: torch.Tensor,
    max_image_height: Optional[int],
    interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """Shrink tall images so height ≤ max_image_height (aspect preserved, no upscale)."""
    if max_image_height is None or max_image_height <= 0:
        return img
    _, h, w = img.shape
    if h <= max_image_height:
        return img
    new_w = max(1, int(round(w * max_image_height / h)))
    return T.functional.resize(
        img, [int(max_image_height), new_w],
        interpolation=interpolation, antialias=True,
    )


def standardize_per_image(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image z-score normalisation: ``(img - mean) / std``."""
    mean = img.mean()
    std = img.std().clamp_min(eps)
    return (img - mean) / std


def pad_to_patch_multiple(
    img: torch.Tensor, pixel_patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zero-pad ``[C, H, W]`` on right / bottom so both spatial sides are
    multiples of *pixel_patch_size*.

    ``pixel_patch_size = patch_size * cnn_effective_stride``.  Padding to this
    multiple ensures the image is divisible by ``cnn_effective_stride`` (for the
    CNN) *and* the resulting CNN feature map is divisible by ``patch_size``
    (for PatchEmbed).

    Returns ``(padded, pad_mask)`` where ``pad_mask`` is ``[1, H', W']``
    boolean (True on padded pixels).
    """
    if img.dim() != 3:
        raise ValueError(f"pad_to_patch_multiple expects [C, H, W], got {tuple(img.shape)}")
    _, h, w = img.shape
    H = math.ceil(h / pixel_patch_size) * pixel_patch_size
    W = math.ceil(w / pixel_patch_size) * pixel_patch_size
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
    img = apply_max_height_shrink(img, image_height, interpolation=interpolation)
    th, tw = int(image_height), int(image_width)
    _, h, w = img.shape
    scale = min(th / h, tw / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = T.functional.resize(img, [new_h, new_w], interpolation=interpolation, antialias=True)
    pad_h, pad_w = th - new_h, tw - new_w
    pad_top, pad_left = pad_h // 2, pad_w // 2
    pad_bottom, pad_right = pad_h - pad_top, pad_w - pad_left
    padded = T.functional.pad(resized, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
    pad_mask = torch.ones(1, th, tw, dtype=torch.bool)
    pad_mask[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = False
    return padded, pad_mask


# =====================================================================
# Augmentation pipeline
# =====================================================================

class MAEAugmentation(nn.Module):
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
    if isinstance(sample, dict):
        image = sample["image"]
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
    batch: List[torch.Tensor], pixel_patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate variable-shape ``[1, H_i, W_i]`` images into a single batch.

    Pads to ``(max_h, max_w)`` rounded **up** to a multiple of
    ``pixel_patch_size = patch_size * cnn_effective_stride``.  This ensures:

    1. The CNN can downsample by ``cnn_effective_stride`` without remainder.
    2. The resulting CNN feature map is divisible by ``patch_size`` for
       :class:`PatchEmbed`.

    Returns ``(images, pad_masks)`` — shapes ``[B, C, H', W']`` and
    ``[B, 1, H', W']`` boolean.
    """
    if len(batch) == 0:
        raise ValueError("mae_pad_collate received an empty batch.")
    C = batch[0].shape[0]
    max_h = max(int(img.shape[-2]) for img in batch)
    max_w = max(int(img.shape[-1]) for img in batch)
    H = math.ceil(max_h / pixel_patch_size) * pixel_patch_size
    W = math.ceil(max_w / pixel_patch_size) * pixel_patch_size
    B = len(batch)
    images = torch.zeros(B, C, H, W, dtype=batch[0].dtype)
    pad_masks = torch.ones(B, 1, H, W, dtype=torch.bool)
    for i, img in enumerate(batch):
        _, h, w = img.shape
        images[i, :, :h, :w] = img
        pad_masks[i, :, :h, :w] = False
    return images, pad_masks


# =====================================================================
# Wrapped dataset
# =====================================================================

class _MAEDatasetWrapper(Dataset):
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
            scale = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]
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
    model: CNNMaskedAutoencoderViT,
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
        images    = images.to(device, non_blocking=True)
        pad_masks = pad_masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss, _, _ = model(images, mask_ratio=mask_ratio, pad_mask=pad_masks)

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
    model: CNNMaskedAutoencoderViT,
    dataloader: DataLoader,
    device: torch.device,
    mask_ratio: float,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for images, pad_masks in dataloader:
        images    = images.to(device, non_blocking=True)
        pad_masks = pad_masks.to(device, non_blocking=True)
        loss, _, _ = model(images, mask_ratio=mask_ratio, pad_mask=pad_masks)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def _save_train_transform_previews(
    train_dataset: Dataset,
    pixel_patch_size: int,
    out_dir: str,
    epoch_1based: int,
    num_samples: int,
) -> None:
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
        padded, pad_mask = pad_to_patch_multiple(image, pixel_patch_size)
        im = padded.detach().float().cpu().numpy().squeeze(0)
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
        f"CNN-MAE training pipeline — after transforms (epoch {epoch_1based})",
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
    model: CNNMaskedAutoencoderViT,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    path: str,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    model_config = {
        "max_image_height": model.max_image_height,
        "patch_size": model.patch_size,
        "in_channels": model.in_channels,
        "embed_dim": model.embed_dim,
        "depth": model.depth,
        "num_heads": model.num_heads,
        "decoder_embed_dim": model.decoder_embed_dim,
        "decoder_depth": model.decoder_depth,
        "decoder_num_heads": model.decoder_num_heads,
        "mlp_ratio": model.mlp_ratio,
        "norm_pix_loss": model.norm_pix_loss,
        "clip_pixel_pred": model.clip_pixel_pred,
        "decoder_pred_num_layers": model.decoder_pred_num_layers,
        "decoder_pred_hidden_dim": model.decoder_pred_hidden_dim,
        "l1_loss_weight": model.l1_loss_weight,
        "l2_loss_weight": model.l2_loss_weight,
        "fft_loss_weight": model.fft_loss_weight,
        # CNN config
        "cnn_layer_channels": model.cnn_layer_channels,
        "cnn_kernel_sizes": model.cnn_kernel_sizes,
        "cnn_strides": model.cnn_strides,
    }

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_config": model_config,
    }, path)


def load_checkpoint(
    path: str,
    device: Optional[torch.device] = None,
) -> Tuple[CNNMaskedAutoencoderViT, Dict[str, Any]]:
    """
    Load a CNN + MAE ViT from a training checkpoint.

    Returns:
        model: Loaded ``CNNMaskedAutoencoderViT`` in eval mode.
        info:  Dict with ``epoch``, ``train_loss``, ``val_loss``, etc.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location="cpu")
    config = dict(ckpt["model_config"])

    # Normalise legacy / missing keys
    config.pop("pixel_min", None)
    config.pop("pixel_max", None)
    config.setdefault("l1_loss_weight", 0.0)
    config.setdefault("l2_loss_weight", 1.0)
    config.setdefault("fft_loss_weight", 0.0)
    config.setdefault("cnn_layer_channels", None)
    config.setdefault("cnn_kernel_sizes", 3)
    config.setdefault("cnn_strides", 1)

    if "max_image_height" not in config:
        if "image_height" in config:
            config["max_image_height"] = int(config.pop("image_height"))
        elif "image_size" in config:
            sz = config.pop("image_size")
            config["max_image_height"] = int(sz) if isinstance(sz, int) else int(max(sz))
    config.pop("image_width", None)
    config.pop("image_size", None)

    model = create_cnn_mae_vit(**config)
    bad = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if bad.missing_keys or bad.unexpected_keys:
        import warnings
        warnings.warn(
            f"Checkpoint loaded with strict=False: "
            f"{len(bad.missing_keys)} missing, {len(bad.unexpected_keys)} unexpected keys.",
            stacklevel=2,
        )
    model.eval()
    model.to(device)

    info = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return model, info


def mae_model_summary(model: CNNMaskedAutoencoderViT) -> Dict[str, Any]:
    """Serializable model architecture / hyperparameters + parameter counts."""
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "max_image_height": model.max_image_height,
        "patch_size": model.patch_size,
        "pixel_patch_size": model.pixel_patch_size,
        "cnn_effective_stride": model.cnn_effective_stride,
        "cnn_layer_channels": model.cnn_layer_channels,
        "cnn_kernel_sizes": model.cnn_kernel_sizes,
        "cnn_strides": model.cnn_strides,
        "in_channels": model.in_channels,
        "embed_dim": model.embed_dim,
        "depth": model.depth,
        "num_heads": model.num_heads,
        "decoder_embed_dim": model.decoder_embed_dim,
        "decoder_depth": model.decoder_depth,
        "decoder_num_heads": model.decoder_num_heads,
        "mlp_ratio": model.mlp_ratio,
        "norm_pix_loss": model.norm_pix_loss,
        "clip_pixel_pred": model.clip_pixel_pred,
        "decoder_pred_num_layers": model.decoder_pred_num_layers,
        "decoder_pred_hidden_dim": model.decoder_pred_hidden_dim,
        "l1_loss_weight": model.l1_loss_weight,
        "l2_loss_weight": model.l2_loss_weight,
        "fft_loss_weight": model.fft_loss_weight,
        "num_parameters": n_params,
        "num_trainable_parameters": n_trainable,
    }


def dump_mae_training_results(
    path: str,
    *,
    model: CNNMaskedAutoencoderViT,
    history: Dict[str, List[float]],
    training_config: Dict[str, Any],
    metric_definitions: Optional[Dict[str, str]] = None,
) -> str:
    path = str(Path(path).resolve())
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model": mae_model_summary(model),
        "training": training_config,
        "metrics_per_epoch": dict(history),
    }
    if metric_definitions is not None:
        payload["metric_definitions"] = metric_definitions

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(f"Training run log written to {path}")
    return path


# =====================================================================
# Visualisation helpers
# =====================================================================

def visualize_reconstruction(
    model: CNNMaskedAutoencoderViT,
    dataset: Dataset,
    num_samples: int = 4,
    mask_ratio: float = 0.75,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    sample_indices: Optional[List[int]] = None,
):
    """
    Show three views per sample: Original | Reconstructed | Masked.

    Reconstruction is at ``model.pixel_patch_size`` granularity so it matches
    the original-pixel patch receptive field (CNN stride × ViT patch size).
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

    pps = model.pixel_patch_size

    for i, idx in enumerate(indices):
        img = val_tf(dataset[int(idx)])
        img, pad_mask_px = pad_to_patch_multiple(img, pps)
        img = img.unsqueeze(0).to(device)
        pad_mask_px = pad_mask_px.unsqueeze(0).to(device)

        with torch.no_grad():
            loss, pred, mask = model(img, mask_ratio=mask_ratio, pad_mask=pad_mask_px)

        patches = model.patchify(img)
        mask_expanded = mask.unsqueeze(-1).expand_as(patches)

        grid_h = img.shape[-2] // pps
        grid_w = img.shape[-1] // pps

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

        orig_np = _to_np(img)
        lo, hi = float(orig_np.min()), float(orig_np.max())

        axes[i, 0].imshow(orig_np, cmap="gray", vmin=lo, vmax=hi)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(_to_np(hybrid_img), cmap="gray", vmin=lo, vmax=hi)
        axes[i, 1].set_title(f"Reconstructed\n(loss={loss.item():.4f})")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(_to_np(masked_img), cmap="gray", vmin=lo, vmax=hi)
        axes[i, 2].set_title(f"Masked ({mask_ratio:.0%})")
        axes[i, 2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()


def dump_mae_training_metrics_artifacts(
    history: Dict[str, List[float]],
    out_dir: str,
    epoch_1based: int,
) -> Tuple[str, str]:
    """Persist training curves: CSV + PNG (loss linear, loss log, LR)."""
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

    fig.suptitle(f"CNN-MAE training metrics (through epoch {n})", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  → MAE metrics CSV: {csv_path}")
    print(f"  → MAE metrics plot: {png_path}")
    return csv_path, png_path


# =====================================================================
# Training plots
# =====================================================================

def save_cnn_mae_training_plots(history: Dict, out_dir: str) -> str:
    """
    Save a PNG with loss, RankME (when available), and LR panels.
    Overwrites ``{out_dir}/training_plots.png`` each call.

    Returns the path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(history.get("train_loss", []))
    if n == 0:
        return ""
    xs = range(1, n + 1)
    has_rankme = any(v is not None for v in history.get("val_rank_me", []))
    ncols = 2 + int(has_rankme)

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    ax = axes[0]
    ax.plot(xs, history["train_loss"], label="train", linewidth=1.5)
    ax.plot(xs, history["val_loss"],   label="val",   linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Reconstruction Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(xs, history["lr"], color="tab:orange", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.set_title("LR Schedule")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(True, alpha=0.3)

    if has_rankme:
        ax = axes[2]
        rm_xs  = [i + 1 for i, v in enumerate(history["val_rank_me"])   if v is not None]
        rm_tr  = [v      for v in history["train_rank_me"] if v is not None]
        rm_val = [v      for v in history["val_rank_me"]   if v is not None]
        ax.plot(rm_xs, rm_tr,  "o-", label="train", linewidth=1.5, markersize=4)
        ax.plot(rm_xs, rm_val, "o-", label="val",   linewidth=1.5, markersize=4)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Effective rank")
        ax.set_title("RankME")
        ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(f"CNN-MAE training — epoch {n}", fontsize=11, y=1.01)
    fig.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, "training_plots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → training plots saved: {path}")
    return path


# =====================================================================
# Main entry point
# =====================================================================

def train_mae(
    dataset: Dataset,
    # --- CNN config ---
    cnn_layer_channels: Optional[List[int]] = None,
    cnn_kernel_sizes: Union[int, List[int]] = 3,
    cnn_strides: Union[int, List[int]] = 1,
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
    # --- device / precision ---
    device: Optional[torch.device] = None,
    use_amp: bool = True,
    # --- resume ---
    resume_from: Optional[str] = None,
    # --- logging ---
    results_json: Optional[str] = None,
    on_epoch_end: Optional[
        Callable[[int, Dict[str, List[float]], CNNMaskedAutoencoderViT], None]
    ] = None,
) -> Tuple[CNNMaskedAutoencoderViT, Dict[str, List[float]]]:
    """
    Train a CNN + MAE ViT from scratch on a grayscale ultrasound dataset.

    CNN configuration
    -----------------
    cnn_layer_channels : list of int, default [32, 64]
        Output channels per CNN layer.  ``len(cnn_layer_channels)`` sets the
        number of convolutional layers before ViT tokenisation.
    cnn_kernel_sizes : int or list of int, default 3
        Kernel size(s).  Single int applies to all layers.
    cnn_strides : int or list of int, default 1
        Stride(s).  Single int applies to all layers.
        ``effective_stride = product(cnn_strides)``; images are padded to
        multiples of ``pixel_patch_size = patch_size * effective_stride``.

    All other arguments are identical to ``embeddings.vit.train.train_mae``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if max_image_height <= 0:
        raise ValueError("max_image_height must be positive.")

    # Compute effective CNN stride before model creation (needed for validation
    # and collate function).
    _layer_channels = list(cnn_layer_channels) if cnn_layer_channels else [32, 64]
    _n = len(_layer_channels)
    _strides = (
        [cnn_strides] * _n if isinstance(cnn_strides, int) else list(cnn_strides)
    )
    cnn_effective_stride = 1
    for s in _strides:
        cnn_effective_stride *= s
    pixel_patch_size = patch_size * cnn_effective_stride

    if max_image_height % pixel_patch_size != 0:
        raise ValueError(
            f"max_image_height must be divisible by pixel_patch_size "
            f"(= patch_size × cnn_effective_stride = {patch_size} × "
            f"{cnn_effective_stride} = {pixel_patch_size}); "
            f"got max_image_height={max_image_height}."
        )
    print(f"Training on: {device}")
    print(
        f"CNN: {_layer_channels} channels, kernels={cnn_kernel_sizes}, "
        f"strides={cnn_strides} → effective_stride={cnn_effective_stride}, "
        f"pixel_patch_size={pixel_patch_size}"
    )

    # ---- split dataset ----
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {n_train} samples | Val: {n_val} samples")

    train_ds = _MAEDatasetWrapper(
        train_ds, MAETrainTransform(max_image_height=max_image_height),
    )
    val_ds = _MAEDatasetWrapper(
        val_ds, MAEValTransform(max_image_height=max_image_height),
    )

    def _collate(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return mae_pad_collate(batch, pixel_patch_size)

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
    model = create_cnn_mae_vit(
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
        cnn_layer_channels=_layer_channels,
        cnn_kernel_sizes=cnn_kernel_sizes,
        cnn_strides=cnn_strides,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(
        p.numel() for n, p in model.named_parameters()
        if not n.startswith("decoder") and not n.startswith("mask_token")
    )
    print(f"Model parameters: {n_params:,} total | {n_enc:,} encoder")

    training_config: Dict[str, Any] = {
        "max_image_height": max_image_height,
        "patch_size": patch_size,
        "pixel_patch_size": pixel_patch_size,
        "cnn_layer_channels": _layer_channels,
        "cnn_kernel_sizes": (
            [cnn_kernel_sizes] * _n if isinstance(cnn_kernel_sizes, int) else list(cnn_kernel_sizes)
        ),
        "cnn_strides": _strides,
        "cnn_effective_stride": cnn_effective_stride,
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
        "decoder_embed_dim": decoder_embed_dim,
        "decoder_depth": decoder_depth,
        "decoder_num_heads": decoder_num_heads,
        "mlp_ratio": mlp_ratio,
        "norm_pix_loss": norm_pix_loss,
        "decoder_pred_num_layers": decoder_pred_num_layers,
        "decoder_pred_hidden_dim": decoder_pred_hidden_dim,
        "l1_loss_weight": l1_loss_weight,
        "l2_loss_weight": l2_loss_weight,
        "fft_loss_weight": fft_loss_weight,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "min_lr": min_lr,
        "mask_ratio": mask_ratio,
        "val_split": val_split,
        "num_workers": num_workers,
        "checkpoint_dir": checkpoint_dir,
        "save_every": save_every,
        "rankme_every": rankme_every,
        "plot_every": plot_every,
        "resume_from": resume_from,
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "device": str(device),
    }

    _use_amp = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if _use_amp else None
    if _use_amp:
        print("Mixed precision (AMP) enabled.")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs, min_lr=min_lr,
    )

    out_json = results_json
    if out_json is None and checkpoint_dir:
        out_json = os.path.join(checkpoint_dir, "mae_training_run.json")

    start_epoch = 0

    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    history: Dict[str, List] = {
        "train_loss": [], "val_loss": [],
        "train_rank_me": [], "val_rank_me": [],
        "lr": [],
    }

    metric_definitions = {
        "train_loss": "Mean MAE reconstruction loss on training batches.",
        "val_loss": "Mean MAE reconstruction loss on validation batches.",
        "train_rank_me": (
            "RankME effective rank of train embeddings (Garrido et al., ICML 2023). "
            "Sanity-check: should track val_rank_me. null on non-triggered epochs."
        ),
        "val_rank_me": (
            "RankME effective rank of val embeddings (Garrido et al., ICML 2023). "
            "null on epochs where rankme_every did not trigger."
        ),
        "lr": "Learning rate before optimizer step.",
    }

    # ---- restore history from existing JSON (enables correct resume) ----
    if start_epoch > 0 and out_json and os.path.isfile(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as _f:
                _saved = json.load(_f).get("metrics_per_epoch", {})
            for key in history:
                if key in _saved and isinstance(_saved[key], list):
                    history[key] = list(_saved[key][:start_epoch])
            print(f"Restored {len(history['train_loss'])} epochs of history from {out_json}")
        except Exception as _e:
            print(f"Warning: could not load history from {out_json}: {_e}")

    if out_json:
        dump_mae_training_results(
            out_json, model=model, history=history,
            training_config=training_config, metric_definitions=metric_definitions,
        )

    for epoch in range(start_epoch, epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, device, mask_ratio, epoch, scaler,
        )
        val_loss = _validate(model, val_loader, device, mask_ratio)

        if rankme_every > 0 and (epoch + 1) % rankme_every == 0:
            train_rank_me = compute_rank_me(collect_embeddings(model, train_loader, device))
            val_rank_me   = compute_rank_me(collect_embeddings(model, val_loader, device))
        else:
            train_rank_me = val_rank_me = None

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rank_me"].append(train_rank_me)
        history["val_rank_me"].append(val_rank_me)
        history["lr"].append(current_lr)

        _rm = (
            f" | rank_me(tr/val)={train_rank_me:.1f}/{val_rank_me:.1f}"
            if val_rank_me is not None else ""
        )
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={current_lr:.2e}"
            + _rm
        )

        if on_epoch_end is not None:
            on_epoch_end(epoch, history, model)

        if checkpoint_dir and (epoch + 1) % save_every == 0:
            path = os.path.join(checkpoint_dir, f"mae_epoch_{epoch + 1}.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, path)
            print(f"  → checkpoint saved: {path}")
            if out_json:
                dump_mae_training_results(
                    out_json, model=model, history=history,
                    training_config=training_config, metric_definitions=metric_definitions,
                )
            if train_transform_preview > 0:
                _save_train_transform_previews(
                    train_ds, pixel_patch_size, checkpoint_dir,
                    epoch + 1, train_transform_preview,
                )

        if checkpoint_dir and plot_every > 0 and (epoch + 1) % plot_every == 0:
            save_cnn_mae_training_plots(history, checkpoint_dir)

    if checkpoint_dir:
        path = os.path.join(checkpoint_dir, "mae_final.pt")
        save_checkpoint(
            model, optimizer, scheduler,
            epochs - 1, history["train_loss"][-1],
            history["val_loss"][-1] if history["val_loss"] else None,
            path,
        )
        print(f"Final checkpoint saved: {path}")

    if out_json:
        dump_mae_training_results(
            out_json, model=model, history=history,
            training_config=training_config, metric_definitions=metric_definitions,
        )

    if checkpoint_dir and plot_every > 0:
        save_cnn_mae_training_plots(history, checkpoint_dir)

    return model, history
