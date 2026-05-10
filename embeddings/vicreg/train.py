"""
VICReg training script for the ultrasound ViT.

Follows the same patterns as ``embeddings/vit/train.py``.

Usage (notebook)::

    from embeddings.vicreg.train import train_vicreg

    model, history = train_vicreg(
        dataset=ds,
        max_image_height=224,
        patch_size=16,
        embed_dim=512,
        depth=8,
        num_heads=8,
        checkpoint_dir="experiments/checkpoints/vicreg/v1",
    )
    # history: train_loss, val_loss + sim/var/cov breakdowns, rank_me, lr per epoch.
"""

import json
import math
import os
import random
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from embeddings.vit.train import (
    CosineWarmupScheduler,
    _MAEDatasetWrapper,
    _extract_image,
    apply_max_height_shrink,
    mae_pad_collate,
    standardize_per_image,
    MAEValTransform,
)
from embeddings.vicreg.model import VICRegModel, create_vicreg, create_vicreg_timm
from embeddings.rank_me import collect_embeddings, compute_rank_me


# =====================================================================
# Augmentation — two independent views per image
# =====================================================================

class _VICRegAugmentation:
    """Stochastic augmentation applied independently to each view."""

    def __init__(
        self,
        flip_p: float = 0.5,
        rotation_degrees: float = 10.0,
        rotation_p: float = 0.3,
        blur_p: float = 0.3,
        jitter_strength: float = 0.2,
        jitter_p: float = 0.5,
    ):
        self.flip = T.RandomHorizontalFlip(p=flip_p)
        self.rotation = T.RandomApply([T.RandomRotation(rotation_degrees)], p=rotation_p)
        self.blur = T.RandomApply(
            [T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=blur_p
        )
        self.jitter_strength = jitter_strength
        self.jitter_p = jitter_p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = self.flip(img)
        img = self.rotation(img)
        img = self.blur(img)
        if random.random() < self.jitter_p:
            # Brightness/contrast jitter on z-scored image.
            scale = 1.0 + random.uniform(-self.jitter_strength, self.jitter_strength)
            img = img * scale
        return img


class VICRegTransform:
    """
    Produces two independently augmented views of the same image.

    Returns ``(view1, view2)`` as ``[1, H, W]`` tensors (same spatial size).
    Shared per-image z-score normalisation is applied before augmentation so
    brightness jitter acts on a standardised signal.
    """

    def __init__(self, max_image_height: int = 224):
        self.max_image_height = int(max_image_height)
        self.aug = _VICRegAugmentation()

    def __call__(self, sample: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        img = _extract_image(sample)
        img = apply_max_height_shrink(img, self.max_image_height)
        img = standardize_per_image(img)
        return self.aug(img), self.aug(img)


class _VICRegDatasetWrapper(Dataset):
    def __init__(self, base: Dataset, transform: VICRegTransform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.transform(self.base[idx])


# =====================================================================
# Collate — variable-size view pairs
# =====================================================================

def _pad_views(
    imgs: List[torch.Tensor], patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    C = imgs[0].shape[0]
    max_h = max(x.shape[1] for x in imgs)
    max_w = max(x.shape[2] for x in imgs)
    H = math.ceil(max_h / patch_size) * patch_size
    W = math.ceil(max_w / patch_size) * patch_size
    B = len(imgs)
    out = torch.zeros(B, C, H, W, dtype=imgs[0].dtype)
    masks = torch.ones(B, 1, H, W, dtype=torch.bool)
    for i, img in enumerate(imgs):
        _, h, w = img.shape
        out[i, :, :h, :w] = img
        masks[i, :, :h, :w] = False
    return out, masks


def vicreg_pad_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]], patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate variable-shape view pairs.

    Returns:
        imgs1, pad_masks1, imgs2, pad_masks2 — each [B, 1, H, W].
    """
    v1 = [item[0] for item in batch]
    v2 = [item[1] for item in batch]
    imgs1, pads1 = _pad_views(v1, patch_size)
    imgs2, pads2 = _pad_views(v2, patch_size)
    return imgs1, pads1, imgs2, pads2


# =====================================================================
# Checkpointing
# =====================================================================

def _encoder_config(model: VICRegModel) -> dict:
    """Serializable config needed to reconstruct the encoder from scratch."""
    from embeddings.timm_encoder import TimmViTEncoder
    enc = model.encoder
    if isinstance(enc, TimmViTEncoder):
        return {
            'encoder_type': 'timm',
            'timm_model': enc.model_name,
            'projector_dim': model.projector_dim,
            'projector_layers': model.projector_layers,
        }
    return {
        'encoder_type': 'mae',
        'patch_size': enc.patch_size,
        'embed_dim': enc.embed_dim,
        'depth': enc.depth,
        'num_heads': enc.num_heads,
        'mlp_ratio': enc.mlp_ratio,
        'max_image_height': enc.max_image_height,
        'projector_dim': model.projector_dim,
        'projector_layers': model.projector_layers,
    }


def save_checkpoint(
    model: VICRegModel,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    path: str,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_config': _encoder_config(model),
    }, path)


def load_checkpoint(
    path: str,
    device: Optional[torch.device] = None,
) -> Tuple[VICRegModel, Dict[str, Any]]:
    """
    Load a VICReg model from a training checkpoint.

    Returns:
        model:  Loaded ``VICRegModel`` in eval mode.
        info:   Dict with epoch, train_loss, val_loss, model_config.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(path, map_location='cpu')
    cfg = ckpt['model_config']
    if cfg.get('encoder_type') == 'timm':
        model = create_vicreg_timm(
            timm_model=cfg['timm_model'],
            projector_dim=cfg['projector_dim'],
            projector_layers=cfg['projector_layers'],
            pretrained=False,  # weights come from the checkpoint
        )
    else:
        model = create_vicreg(
            patch_size=cfg['patch_size'],
            embed_dim=cfg['embed_dim'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            mlp_ratio=cfg['mlp_ratio'],
            projector_dim=cfg['projector_dim'],
            projector_layers=cfg['projector_layers'],
            max_image_height=cfg.get('max_image_height'),
        )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(device)
    info = {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
    return model, info


# =====================================================================
# Results dump
# =====================================================================

def _vicreg_model_summary(model: VICRegModel) -> Dict[str, Any]:
    from embeddings.timm_encoder import TimmViTEncoder
    enc = model.encoder
    n_enc = sum(p.numel() for p in model.encoder_parameters())
    n_proj = sum(p.numel() for p in model.projector.parameters())
    base = {
        'patch_size': enc.patch_size,
        'embed_dim': enc.embed_dim,
        'max_image_height': enc.max_image_height,
        'projector_dim': model.projector_dim,
        'projector_layers': model.projector_layers,
        'num_encoder_parameters': n_enc,
        'num_projector_parameters': n_proj,
        'num_trainable_parameters': n_enc + n_proj,
    }
    if isinstance(enc, TimmViTEncoder):
        base['encoder_type'] = 'timm'
        base['timm_model'] = enc.model_name
    else:
        base['encoder_type'] = 'mae'
        base['depth'] = enc.depth
        base['num_heads'] = enc.num_heads
        base['mlp_ratio'] = enc.mlp_ratio
    return base


def dump_vicreg_training_results(
    path: str,
    *,
    model: VICRegModel,
    history: Dict[str, List],
    training_config: Dict[str, Any],
    metric_definitions: Optional[Dict[str, str]] = None,
) -> str:
    path = str(Path(path).resolve())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        'model': _vicreg_model_summary(model),
        'training': training_config,
        'metrics_per_epoch': dict(history),
    }
    if metric_definitions:
        payload['metric_definitions'] = metric_definitions
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(f"Training run log written to {path}")
    return path


# =====================================================================
# Training plots
# =====================================================================

def save_vicreg_training_plots(history: Dict, out_dir: str) -> str:
    """
    Save a PNG with total loss, loss components, RankME (when available), and LR panels.
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
    has_rankme = any(v is not None for v in history.get('val_rank_me', []))
    ncols = 3 + int(has_rankme)

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    ax = axes[0]
    ax.plot(xs, history['train_loss'], label='train', linewidth=1.5)
    ax.plot(xs, history['val_loss'],   label='val',   linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Total VICReg Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(xs, history['train_loss_sim'], label='sim (inv)',   linewidth=1.5)
    ax.plot(xs, history['train_loss_var'], label='var (hinge)', linewidth=1.5)
    ax.plot(xs, history['train_loss_cov'], label='cov (decor)', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Raw term value')
    ax.set_title('Loss Components (train)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(xs, history['lr'], color='tab:orange', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('LR')
    ax.set_title('LR Schedule')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.grid(True, alpha=0.3)

    if has_rankme:
        ax = axes[3]
        rm_xs  = [i + 1 for i, v in enumerate(history['val_rank_me'])   if v is not None]
        rm_tr  = [v      for v in history['train_rank_me'] if v is not None]
        rm_val = [v      for v in history['val_rank_me']   if v is not None]
        ax.plot(rm_xs, rm_tr,  'o-', label='train', linewidth=1.5, markersize=4)
        ax.plot(rm_xs, rm_val, 'o-', label='val',   linewidth=1.5, markersize=4)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Effective rank')
        ax.set_title('RankME')
        ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(f"VICReg training — epoch {n}", fontsize=11, y=1.01)
    fig.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, 'training_plots.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → training plots saved: {path}")
    return path


# =====================================================================
# Single-epoch routines
# =====================================================================

def _train_one_epoch(
    model: VICRegModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    sim_coeff: float,
    std_coeff: float,
    cov_coeff: float,
    epoch: int,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> Tuple[float, float, float, float]:
    """Returns (mean_total, mean_sim, mean_var, mean_cov) over the epoch."""
    model.train()
    use_amp = scaler is not None
    t_loss = t_sim = t_var = t_cov = 0.0
    n = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [train]", leave=False)
    for imgs1, pads1, imgs2, pads2 in pbar:
        imgs1 = imgs1.to(device, non_blocking=True)
        pads1 = pads1.to(device, non_blocking=True)
        imgs2 = imgs2.to(device, non_blocking=True)
        pads2 = pads2.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            z1, z2 = model(imgs1, pads1, imgs2, pads2)
            loss, sim, var, cov = model.vicreg_loss(z1, z2, sim_coeff, std_coeff, cov_coeff)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(list(model.trainable_parameters()), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.trainable_parameters()), max_norm=1.0)
            optimizer.step()

        t_loss += loss.item(); t_sim += sim; t_var += var; t_cov += cov
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.3f}", sim=f"{sim:.3f}", var=f"{var:.3f}")

    d = max(n, 1)
    return t_loss / d, t_sim / d, t_var / d, t_cov / d


@torch.no_grad()
def _validate(
    model: VICRegModel,
    dataloader: DataLoader,
    device: torch.device,
    sim_coeff: float,
    std_coeff: float,
    cov_coeff: float,
) -> Tuple[float, float, float, float]:
    """Returns (mean_total, mean_sim, mean_var, mean_cov) on the validation set."""
    model.eval()
    t_loss = t_sim = t_var = t_cov = 0.0
    n = 0
    for imgs1, pads1, imgs2, pads2 in dataloader:
        imgs1 = imgs1.to(device, non_blocking=True)
        pads1 = pads1.to(device, non_blocking=True)
        imgs2 = imgs2.to(device, non_blocking=True)
        pads2 = pads2.to(device, non_blocking=True)
        z1, z2 = model(imgs1, pads1, imgs2, pads2)
        loss, sim, var, cov = model.vicreg_loss(z1, z2, sim_coeff, std_coeff, cov_coeff)
        t_loss += loss.item(); t_sim += sim; t_var += var; t_cov += cov
        n += 1
    d = max(n, 1)
    return t_loss / d, t_sim / d, t_var / d, t_cov / d


# =====================================================================
# Main entry point
# =====================================================================

def train_vicreg(
    dataset: Dataset,
    # --- model ---
    max_image_height: int = 224,
    patch_size: int = 16,
    embed_dim: int = 512,
    depth: int = 8,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    projector_dim: int = 2048,
    projector_layers: int = 3,
    # --- VICReg loss weights ---
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    # --- training ---
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 10,
    min_lr: float = 1e-6,
    val_split: float = 0.1,
    num_workers: int = 4,
    # --- checkpointing ---
    checkpoint_dir: Optional[str] = None,
    save_every: int = 10,
    rankme_every: int = 0,
    plot_every: int = 0,
    # --- encoder selection ---
    timm_encoder: Optional[str] = None,
    # --- warm-start (mae encoder only) ---
    mae_init_checkpoint: Optional[str] = None,
    pretrained_encoder: Optional[str] = None,
    # --- device / precision ---
    device: Optional[torch.device] = None,
    use_amp: bool = True,
    # --- resume ---
    resume_from: Optional[str] = None,
    # --- logging ---
    results_json: Optional[str] = None,
    on_epoch_end: Optional[
        Callable[[int, Dict[str, List], VICRegModel], None]
    ] = None,
) -> Tuple[VICRegModel, Dict[str, List]]:
    """
    Train a VICReg model on a grayscale ultrasound dataset.

    Args:
        dataset:          Any ``torch.utils.data.Dataset`` returning
                          grayscale image tensors (or dicts with ``'image'``).
        max_image_height: Cap on per-image height (aspect-preserved shrink, no upscale).
        patch_size:       Patch side length.
        embed_dim:        Encoder embedding dimension.
        depth:            Encoder depth (number of transformer blocks).
        num_heads:        Encoder attention heads.
        mlp_ratio:        MLP expansion ratio.
        projector_dim:    Output dimension of the MLP projector.
        projector_layers: Number of layers in the projector.
        sim_coeff:        Invariance loss weight (λ in the paper, default 25).
        std_coeff:        Variance loss weight (μ in the paper, default 25).
        cov_coeff:        Covariance loss weight (ν in the paper, default 1).
        epochs:           Total training epochs.
        batch_size:       Batch size. Larger batches (≥256) improve cov loss estimates.
        lr:               Peak learning rate.
        weight_decay:     AdamW weight decay.
        warmup_epochs:    Linear warmup epochs.
        min_lr:           Minimum LR after cosine annealing.
        val_split:        Fraction of data held out for validation.
        num_workers:      DataLoader workers.
        checkpoint_dir:   Where to save checkpoints.  ``None`` = no saving.
        save_every:       Save a checkpoint every N epochs.
        rankme_every:     Compute RankME on train and val every N epochs (0 = off).
        mae_init_checkpoint: Path to an MAE checkpoint to warm-start the encoder.
        device:           Training device (auto-detected when None).
        use_amp:          Enable AMP mixed precision.
        resume_from:      Path to a VICReg checkpoint to resume training.
        results_json:     Path for JSON training log. Defaults to
                          ``{checkpoint_dir}/vicreg_training_run.json``.
        on_epoch_end:     Optional callback ``(epoch, history, model)`` after each epoch.

    Returns:
        model:   Trained ``VICRegModel``.
        history: Per-epoch metrics dict.
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

    # ---- pair loaders (for VICReg loss) ----
    train_pair_ds = _VICRegDatasetWrapper(train_ds, VICRegTransform(max_image_height))
    val_pair_ds   = _VICRegDatasetWrapper(val_ds,   VICRegTransform(max_image_height))

    _collate_pair = partial(vicreg_pad_collate, patch_size=patch_size)
    _persistent = num_workers > 0
    train_loader = DataLoader(
        train_pair_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=_collate_pair, persistent_workers=_persistent,
    )
    val_loader = DataLoader(
        val_pair_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=_collate_pair, persistent_workers=_persistent,
    )

    # ---- single-view loaders (for RankME) ----
    _collate_single = partial(mae_pad_collate, patch_size=patch_size)
    _val_transform = MAEValTransform(max_image_height=max_image_height)
    embed_train_ds = _MAEDatasetWrapper(train_ds, _val_transform)
    embed_val_ds   = _MAEDatasetWrapper(val_ds,   _val_transform)
    embed_train_loader = DataLoader(
        embed_train_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=_collate_single, persistent_workers=_persistent,
    )
    embed_val_loader = DataLoader(
        embed_val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=_collate_single, persistent_workers=_persistent,
    )

    # ---- model ----
    if timm_encoder is not None:
        model = create_vicreg_timm(
            timm_model=timm_encoder,
            projector_dim=projector_dim,
            projector_layers=projector_layers,
        ).to(device)
        print(f"Using timm encoder: {timm_encoder}")
    else:
        model = create_vicreg(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            projector_dim=projector_dim,
            projector_layers=projector_layers,
            max_image_height=max_image_height,
        ).to(device)

    if mae_init_checkpoint is not None:
        if timm_encoder is not None:
            print("Warning: mae_init_checkpoint is ignored when timm_encoder is set.")
        else:
            ckpt = torch.load(mae_init_checkpoint, map_location='cpu')
            missing, unexpected = model.encoder.load_state_dict(
                ckpt['model_state_dict'], strict=False
            )
            print(
                f"MAE encoder warm-start: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys (decoder keys expected)."
            )

    if pretrained_encoder is not None and mae_init_checkpoint is None and resume_from is None:
        if timm_encoder is not None:
            print("Warning: pretrained_encoder is ignored when timm_encoder is set.")
        else:
            from embeddings.pretrained_loader import load_pretrained_vit_encoder
            _pt_info = load_pretrained_vit_encoder(model.encoder, pretrained_encoder)
            print(f"Pretrained init: {_pt_info['summary']}")

    n_enc  = sum(p.numel() for p in model.encoder_parameters())
    n_proj = sum(p.numel() for p in model.projector.parameters())
    print(f"Parameters: {n_enc:,} encoder | {n_proj:,} projector | "
          f"{n_enc + n_proj:,} total trainable")

    training_config: Dict[str, Any] = {
        'max_image_height': max_image_height,
        'patch_size': patch_size,
        'embed_dim': embed_dim,
        'depth': depth,
        'num_heads': num_heads,
        'mlp_ratio': mlp_ratio,
        'projector_dim': projector_dim,
        'projector_layers': projector_layers,
        'sim_coeff': sim_coeff,
        'std_coeff': std_coeff,
        'cov_coeff': cov_coeff,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'warmup_epochs': warmup_epochs,
        'min_lr': min_lr,
        'val_split': val_split,
        'num_workers': num_workers,
        'checkpoint_dir': checkpoint_dir,
        'save_every': save_every,
        'rankme_every': rankme_every,
        'plot_every': plot_every,
        'timm_encoder': timm_encoder,
        'mae_init_checkpoint': mae_init_checkpoint,
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
    optimizer = optim.AdamW(
        list(model.trainable_parameters()), lr=lr, weight_decay=weight_decay
    )
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs, min_lr=min_lr,
    )

    out_json = results_json
    if out_json is None and checkpoint_dir:
        out_json = os.path.join(checkpoint_dir, 'vicreg_training_run.json')

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
        'train_loss': [], 'train_loss_sim': [], 'train_loss_var': [], 'train_loss_cov': [],
        'val_loss':   [], 'val_loss_sim':   [], 'val_loss_var':   [], 'val_loss_cov':   [],
        'train_rank_me': [], 'val_rank_me': [],
        'lr': [],
    }

    metric_definitions = {
        'train_loss':     'Total VICReg loss on training batches.',
        'train_loss_sim': 'Invariance (MSE between two views) term, training.',
        'train_loss_var': 'Variance hinge term (keeps per-dim std >= 1), training.',
        'train_loss_cov': 'Covariance regularization term (decorrelates dims), training.',
        'val_loss':       'Total VICReg loss on validation set.',
        'val_loss_sim':   'Invariance term, validation.',
        'val_loss_var':   'Variance hinge term, validation.',
        'val_loss_cov':   'Covariance term, validation.',
        'train_rank_me': (
            'RankME effective rank of train embeddings (Garrido et al., ICML 2023). '
            'null on non-triggered epochs.'
        ),
        'val_rank_me': (
            'RankME effective rank of val embeddings. '
            'null on epochs where rankme_every did not trigger.'
        ),
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
        dump_vicreg_training_results(
            out_json, model=model, history=history,
            training_config=training_config, metric_definitions=metric_definitions,
        )

    # ---- training loop ----
    for epoch in range(start_epoch, epochs):
        current_lr = optimizer.param_groups[0]['lr']

        tr_loss, tr_sim, tr_var, tr_cov = _train_one_epoch(
            model, train_loader, optimizer, device,
            sim_coeff, std_coeff, cov_coeff, epoch, scaler,
        )
        val_loss, val_sim, val_var, val_cov = _validate(
            model, val_loader, device, sim_coeff, std_coeff, cov_coeff,
        )

        if rankme_every > 0 and (epoch + 1) % rankme_every == 0:
            train_rank_me = compute_rank_me(collect_embeddings(model, embed_train_loader, device))
            val_rank_me   = compute_rank_me(collect_embeddings(model, embed_val_loader, device))
        else:
            train_rank_me = val_rank_me = None

        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_loss_sim'].append(tr_sim)
        history['train_loss_var'].append(tr_var)
        history['train_loss_cov'].append(tr_cov)
        history['val_loss'].append(val_loss)
        history['val_loss_sim'].append(val_sim)
        history['val_loss_var'].append(val_var)
        history['val_loss_cov'].append(val_cov)
        history['train_rank_me'].append(train_rank_me)
        history['val_rank_me'].append(val_rank_me)
        history['lr'].append(current_lr)

        _rm = (
            f" | rank_me(tr/val)={train_rank_me:.1f}/{val_rank_me:.1f}"
            if val_rank_me is not None else ""
        )
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss={tr_loss:.4f}/{val_loss:.4f} | "
            f"sim={tr_sim:.3f} var={tr_var:.3f} cov={tr_cov:.3f} | "
            f"lr={current_lr:.2e}"
            + _rm
        )

        if on_epoch_end is not None:
            on_epoch_end(epoch, history, model)

        if checkpoint_dir and (epoch + 1) % save_every == 0:
            path = os.path.join(checkpoint_dir, f"vicreg_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, tr_loss, val_loss, path)
            print(f"  → checkpoint saved: {path}")
            if out_json:
                dump_vicreg_training_results(
                    out_json, model=model, history=history,
                    training_config=training_config, metric_definitions=metric_definitions,
                )

        if checkpoint_dir and plot_every > 0 and (epoch + 1) % plot_every == 0:
            save_vicreg_training_plots(history, checkpoint_dir)

    # ---- final checkpoint ----
    if checkpoint_dir:
        path = os.path.join(checkpoint_dir, "vicreg_final.pt")
        save_checkpoint(
            model, optimizer, scheduler,
            epochs - 1,
            history['train_loss'][-1],
            history['val_loss'][-1] if history['val_loss'] else None,
            path,
        )
        print(f"Final checkpoint saved: {path}")

    if out_json:
        dump_vicreg_training_results(
            out_json, model=model, history=history,
            training_config=training_config, metric_definitions=metric_definitions,
        )

    if checkpoint_dir and plot_every > 0:
        save_vicreg_training_plots(history, checkpoint_dir)

    return model, history
