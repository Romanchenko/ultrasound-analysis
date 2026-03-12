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

The dataset can be *any* ``torch.utils.data.Dataset`` whose ``__getitem__``
returns either:

* a dict with an ``'image'`` key containing a ``[1, H, W]`` grayscale tensor, or
* a plain ``[1, H, W]`` grayscale tensor.

Images are resized to ``image_size × image_size`` on the fly with
ultrasound-appropriate augmentations (random crop, flip, small rotation).
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from tqdm import tqdm

from embeddings.vit.model import MaskedAutoencoderViT, create_mae_vit


# =====================================================================
# Augmentation pipeline
# =====================================================================

class MAEAugmentation(nn.Module):
    """
    Ultrasound-appropriate augmentations for MAE pre-training.

    All operations work on single-channel ``[1, H, W]`` tensors.
    The output is always ``[1, image_size, image_size]``.
    """

    def __init__(self, image_size: int = 224):
        super().__init__()
        self.image_size = image_size
        self.transform = T.Compose([
            T.RandomResizedCrop(
                image_size,
                scale=(0.6, 1.0),
                ratio=(0.75, 1.333),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.RandomRotation(degrees=15)], p=0.3),
            T.RandomApply([
                T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            ], p=0.2),
        ])

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: [1, H, W] or [H, W] grayscale tensor in [0, 1].
        Returns:
            [1, image_size, image_size]
        """
        if img.dim() == 2:
            img = img.unsqueeze(0)
        return self.transform(img)


class MAETrainTransform:
    """
    Full transform applied to raw dataset samples before feeding to MAE.

    Handles both dict and tensor samples from the dataset, ensures
    grayscale ``[1, H, W]`` output, and applies augmentation.
    """

    def __init__(self, image_size: int = 224):
        self.augmentation = MAEAugmentation(image_size)
        self.resize = T.Resize(
            (image_size, image_size),
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )
        self.image_size = image_size

    def __call__(self, sample: Any) -> torch.Tensor:
        # Extract image tensor
        if isinstance(sample, dict):
            image = sample['image']
        elif isinstance(sample, (tuple, list)):
            image = sample[0]
        else:
            image = sample

        if not isinstance(image, torch.Tensor):
            image = T.ToTensor()(image)

        # Ensure single channel
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if image.size(0) > 1:
            # Convert to grayscale by averaging channels
            image = image.mean(dim=0, keepdim=True)

        # Apply augmentation (includes resize)
        image = self.augmentation(image)

        return image


class MAEValTransform:
    """Validation transform: just resize, no augmentations."""

    def __init__(self, image_size: int = 224):
        self.resize = T.Resize(
            (image_size, image_size),
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )
        self.image_size = image_size

    def __call__(self, sample: Any) -> torch.Tensor:
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

        image = self.resize(image)
        return image


# =====================================================================
# Wrapped dataset (applies MAE transform on top of base dataset)
# =====================================================================

class _MAEDatasetWrapper(Dataset):
    """Wraps any dataset and applies the MAE transform."""

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
    for batch in pbar:
        batch = batch.to(device)
        loss, _, _ = model(batch, mask_ratio=mask_ratio)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
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

    for batch in dataloader:
        batch = batch.to(device)
        loss, _, _ = model(batch, mask_ratio=mask_ratio)
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

    config = ckpt['model_config']
    model = create_mae_vit(**config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(device)

    info = {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
    return model, info


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
):
    """
    Show original / masked / reconstructed images side by side.

    Works in both Jupyter notebooks and script environments.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Prepare a validation transform (just resize, no augmentation)
    val_tf = MAEValTransform(model.image_size)

    indices = torch.randperm(len(dataset))[:num_samples]

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    for i, idx in enumerate(indices):
        img = val_tf(dataset[int(idx)]).unsqueeze(0).to(device)  # [1,1,H,W]

        with torch.no_grad():
            loss, pred, mask = model(img, mask_ratio=mask_ratio)

        # Reconstruct full image from predictions
        pred_img = model.unpatchify(pred)                  # [1, C, H, W]

        # Build a "masked input" visualisation
        patches = model.patchify(img)                      # [1, N, p*p*C]
        # Zero out masked patches
        mask_expanded = mask.unsqueeze(-1).expand_as(patches)
        masked_patches = patches * (1 - mask_expanded)
        masked_img = model.unpatchify(masked_patches)

        def _to_np(t):
            return t.squeeze().cpu().numpy().clip(0, 1)

        axes[i, 0].imshow(_to_np(img), cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(_to_np(masked_img), cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Masked ({mask_ratio:.0%})')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(_to_np(pred_img), cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Reconstructed (loss={loss.item():.4f})')
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

    Returns:
        model:   Trained ``MaskedAutoencoderViT``.
        history: Dict with ``'train_loss'`` and ``'val_loss'`` lists.
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
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(
        p.numel() for n, p in model.named_parameters()
        if not n.startswith('decoder') and not n.startswith('mask_token')
    )
    print(f"Model parameters: {n_params:,} total | {n_enc:,} encoder")

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

    return model, history

