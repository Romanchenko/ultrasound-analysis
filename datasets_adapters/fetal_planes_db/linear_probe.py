"""
Linear probe evaluation: train a classification head on frozen encoder embeddings.

Evaluates any pretrained encoder on the fetal_planes_db (Plane, Brain_plane)
classification task. The encoder is frozen; only a linear head is trained.

Images are consumed at **variable** spatial size: each sample is shrunk to at
most ``max_image_height`` (aspect preserved, no up-scaling), and the batch is
zero-padded (right/bottom) to a common ``(H, W)`` that is divisible by
``patch_size`` via :func:`pad_classify_collate`. A boolean ``pad_mask`` is
produced per sample and forwarded to ``encoder.encode`` when supported.

Usage::

    from datasets_adapters.fetal_planes_db.linear_probe import train_linear_probe

    # With MAE encoder
    from embeddings.vit.train import load_checkpoint
    encoder, _ = load_checkpoint("checkpoints/mae_final.pt")

    history = train_linear_probe(
        encoder=encoder,
        embed_dim=encoder.embed_dim,
        dataset_root="path/to/FETAL_PLANES_DB",
        max_image_height=512,
        epochs=50,
    )
"""

import inspect
import math
import os
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .fpd_dataset import FetalPlanesDBDataset


def _balanced_accuracy_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> float:
    """Macro-averaged recall (sklearn ``balanced_accuracy_score``)."""
    from sklearn.metrics import balanced_accuracy_score

    if len(y_true) == 0:
        return 0.0
    return float(
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )


def plot_linear_probe_history(
    history: Dict[str, List[float]],
    *,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot loss, accuracy, and balanced accuracy per epoch (from ``train_linear_probe`` history).

    Args:
        history: Dict with ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``,
                  ``train_bal_acc``, ``val_bal_acc`` (fractions in ``[0, 1]``).
        save_path: If set, save figure to this path.
        show: If True, call ``plt.show()`` (e.g. in notebooks).
    """
    import matplotlib.pyplot as plt

    n = len(history["train_loss"])
    epochs = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(epochs, history["train_loss"], label="train", color="#1f77b4")
    axes[0].plot(epochs, history["val_loss"], label="val", color="#ff7f0e")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [100 * x for x in history["train_acc"]], label="train", color="#2ca02c")
    axes[1].plot(epochs, [100 * x for x in history["val_acc"]], label="val", color="#d62728")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, [100 * x for x in history["train_bal_acc"]], label="train (bal.)", color="#9467bd")
    axes[2].plot(epochs, [100 * x for x in history["val_bal_acc"]], label="val (bal.)", color="#8c564b")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Balanced accuracy (%)")
    axes[2].set_title("Balanced accuracy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        parent = os.path.dirname(os.path.abspath(save_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# =====================================================================
# Linear probe model
# =====================================================================

class LinearProbe(nn.Module):
    """
    Frozen encoder + trainable linear classification head.

    Encoder can be any module with:
      - ``.encode(imgs)`` or ``.encode(imgs, pad_mask=...)`` -> ``[B, embed_dim]``, or
      - ``.forward(imgs)`` -> ``[B, embed_dim]`` (embedding as direct output)

    The encoder is kept in eval mode and its parameters are frozen. When the
    encoder's ``.encode`` signature accepts a ``pad_mask`` kwarg, the probe
    forwards the pixel pad mask produced by :func:`pad_classify_collate`.

    Features are passed through a non-affine BatchNorm before the linear head
    (per He et al. 2022 §A.2, "Linear Probing"): frozen MAE features have
    strongly non-uniform per-dim scales, and a parameter-free BN typically
    lifts linear-probe accuracy several points.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        num_classes: int,
    ):
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
                self._encode_supports_pad_mask = False

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


# =====================================================================
# Transforms & collate (variable-shape pipeline)
# =====================================================================

def _apply_max_height_shrink(
    img: torch.Tensor, max_image_height: Optional[int],
) -> torch.Tensor:
    """Shrink the image so its height is at most *max_image_height* (aspect
    preserved). Never upscales. ``None`` / non-positive values are a no-op."""
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
    """Per-image z-score normalisation. Mirror of
    :func:`embeddings.vit.train.standardize_per_image` so the probe sees the
    same pixel distribution the encoder was trained on."""
    mean = img.mean()
    std = img.std().clamp_min(eps)
    return (img - mean) / std


def _default_image_transform(
    max_image_height: int, standardize: bool = True,
) -> Callable[[dict], torch.Tensor]:
    """Build a transform: normalise to ``[1, H, W]`` grayscale, shrink if the
    height exceeds *max_image_height* (aspect preserved, no upscale), and
    apply per-image z-score so the probe's pixel distribution matches the
    MAE pre-training one."""

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


def pad_classify_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]], patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate variable-shape ``[1, H_i, W_i]`` images + labels into a single batch.

    Images are zero-padded (right and bottom only) to ``(max_h, max_w)``, each
    rounded **up** to a multiple of ``patch_size``. Returns
    ``(images, pad_masks, labels)``:

    * ``images``:    ``[B, C, H', W']`` float
    * ``pad_masks``: ``[B, 1, H', W']`` boolean, True where the pixel is padding.
    * ``labels``:    ``[B]`` long tensor.
    """
    if len(batch) == 0:
        raise ValueError("pad_classify_collate received an empty batch.")
    imgs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    C = imgs[0].shape[0]
    max_h = max(int(img.shape[-2]) for img in imgs)
    max_w = max(int(img.shape[-1]) for img in imgs)
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


# =====================================================================
# Dataset wrapper
# =====================================================================

class _ClassificationDatasetWrapper(Dataset):
    """Wraps FetalPlanesDBDataset and applies image transform."""

    def __init__(self, base: FetalPlanesDBDataset, transform: Callable):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.base[idx]
        image = self.transform(item)
        return image, item["label_idx"]


# =====================================================================
# Class mapping
# =====================================================================

def _get_class_mapping(dataset: FetalPlanesDBDataset) -> Dict[str, int]:
    """Build class_to_idx from dataset labels (Plane, Brain_plane) composite."""
    composites = [label["composite"] for label in dataset.labels]
    unique_classes = sorted(set(composites))
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    return class_to_idx


def _infer_patch_size(encoder: nn.Module, default: int = 16) -> int:
    """Read ``patch_size`` off the encoder (MAE-style); fall back to *default*."""
    ps = getattr(encoder, "patch_size", None)
    if ps is None:
        ps = getattr(encoder, "patch_embed", None)
        ps = getattr(ps, "patch_size", None) if ps is not None else None
    try:
        return int(ps) if ps is not None else int(default)
    except (TypeError, ValueError):
        return int(default)


def _warn_legacy_kwargs(**kwargs: Any) -> None:
    """Warn on removed/renamed keyword arguments (``image_size``, ``target_size``)."""
    for name, val in kwargs.items():
        if val is None:
            continue
        warnings.warn(
            f"train_linear_probe({name}=...) is deprecated and ignored: the "
            "linear probe now consumes variable-shape images (shrunk to "
            "max_image_height, then padded per-batch to a patch_size multiple).",
            DeprecationWarning, stacklevel=3,
        )


# =====================================================================
# Training
# =====================================================================

def train_linear_probe(
    encoder: nn.Module,
    embed_dim: int,
    dataset_root: str,
    *,
    images_dir: str = "Images",
    csv_file: str = "FETAL_PLANES_DB_data.csv",
    max_image_height: int = 224,
    patch_size: Optional[int] = None,
    image_transform: Optional[Callable[[dict], torch.Tensor]] = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[str] = None,
    save_every: int = 10,
    show_plot: bool = False,
    # Legacy / deprecated:
    image_size: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, List[float]]:
    """
    Train a linear classification head on frozen encoder embeddings for (Plane, Brain_plane).

    Args:
        encoder: Pretrained encoder (any module with .encode(x[, pad_mask=...])
                 or callable returning ``[B, embed_dim]``). Will be frozen.
        embed_dim: Output dimension of the encoder embeddings.
        dataset_root: Root directory of FETAL_PLANES_DB.
        images_dir: Name of images subdirectory.
        csv_file: Name of CSV metadata file.
        max_image_height: Cap on image height. Images taller than this are
                          shrunk (aspect preserved); shorter ones are left
                          alone. Never upscales.
        patch_size: Patch size used for batch padding. If ``None``, read from
                    ``encoder.patch_size`` (falls back to 16).
        image_transform: Optional callable taking a sample dict and returning
                         an image tensor ``[1, H, W]`` (variable size). If
                         ``None``, uses the default shrink-only transform.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate for the linear head.
        weight_decay: Weight decay for the linear head. Default ``0.0`` —
                      applying WD to a linear classifier over frozen features
                      typically hurts (MAE §A.2).
        num_workers: DataLoader workers.
        device: Device (auto-detect if None).
        checkpoint_dir: Where to save checkpoints (None = no saving).
        save_every: Save checkpoint every N epochs.
        show_plot: If True, display the loss/accuracy/balanced-accuracy figure.
        image_size: **Deprecated** — ignored. Use ``max_image_height``.
        target_size: **Deprecated** — ignored. Aspect ratio is preserved and
                     images are no longer forced into a fixed canvas.

    Returns:
        history: Dict with ``train_loss``, ``train_acc``, ``train_bal_acc``,
                 ``val_loss``, ``val_acc``, ``val_bal_acc`` lists (accuracies
                 as fractions in ``[0, 1]``).
    """
    _warn_legacy_kwargs(image_size=image_size, target_size=target_size)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if patch_size is None:
        patch_size = _infer_patch_size(encoder)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    transform = image_transform or _default_image_transform(max_image_height)
    collate_fn = partial(pad_classify_collate, patch_size=int(patch_size))

    scout = FetalPlanesDBDataset(
        root=dataset_root,
        images_dir=images_dir,
        csv_file=csv_file,
        train=True,
    )
    class_to_idx = _get_class_mapping(scout)
    num_classes = len(class_to_idx)

    train_base = FetalPlanesDBDataset(
        root=dataset_root,
        images_dir=images_dir,
        csv_file=csv_file,
        train=True,
        class_to_idx=class_to_idx,
    )
    val_base = FetalPlanesDBDataset(
        root=dataset_root,
        images_dir=images_dir,
        csv_file=csv_file,
        train=False,
        class_to_idx=class_to_idx,
    )

    train_ds = _ClassificationDatasetWrapper(train_base, transform)
    val_ds = _ClassificationDatasetWrapper(val_base, transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = LinearProbe(encoder, embed_dim, num_classes).to(device)
    # Only the linear head is trainable (feat_bn has no learnable parameters).
    optimizer = optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "train_bal_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_bal_acc": [],
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_labels_chunks: List[torch.Tensor] = []
        train_preds_chunks: List[torch.Tensor] = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False)

        for images, pad_masks, labels in pbar:
            images = images.to(device)
            pad_masks = pad_masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images, pad_mask=pad_masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)
            train_labels_chunks.append(labels.detach().cpu())
            train_preds_chunks.append(pred.detach().cpu())
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*train_correct/train_total:.2f}%")

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        if train_labels_chunks:
            train_y = torch.cat(train_labels_chunks).numpy()
            train_yhat = torch.cat(train_preds_chunks).numpy()
            train_bal_acc = _balanced_accuracy_numpy(train_y, train_yhat, num_classes)
        else:
            train_bal_acc = 0.0

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels_chunks: List[torch.Tensor] = []
        val_preds_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for images, pad_masks, labels in val_loader:
                images = images.to(device)
                pad_masks = pad_masks.to(device)
                labels = labels.to(device)
                logits = model(images, pad_mask=pad_masks)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                pred = logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
                val_labels_chunks.append(labels.detach().cpu())
                val_preds_chunks.append(pred.detach().cpu())

        val_loss /= len(val_loader) if len(val_loader) > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        if val_labels_chunks:
            val_y = torch.cat(val_labels_chunks).numpy()
            val_yhat = torch.cat(val_preds_chunks).numpy()
            val_bal_acc = _balanced_accuracy_numpy(val_y, val_yhat, num_classes)
        else:
            val_bal_acc = 0.0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_bal_acc"].append(train_bal_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_bal_acc"].append(val_bal_acc)

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={100*train_acc:.2f}% train_bal_acc={100*train_bal_acc:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={100*val_acc:.2f}% val_bal_acc={100*val_bal_acc:.2f}%"
        )

        if checkpoint_dir and (epoch + 1) % save_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f"linear_probe_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_to_idx": class_to_idx,
                "embed_dim": embed_dim,
                "max_image_height": int(max_image_height),
                "patch_size": int(patch_size),
                "history": history,
            }, path)
            print(f"  -> checkpoint saved: {path}")

    plot_path = None
    if checkpoint_dir:
        plot_path = os.path.join(checkpoint_dir, "linear_probe_curves.png")
    if plot_path or show_plot:
        plot_linear_probe_history(history, save_path=plot_path, show=show_plot)

    return history


# =====================================================================
# Confusion matrix (after training or from checkpoint)
# =====================================================================


def load_linear_probe_from_checkpoint(
    checkpoint_path: str,
    encoder: nn.Module,
    embed_dim: int,
    device: torch.device,
) -> Tuple[LinearProbe, Dict[str, int]]:
    """
    Load a saved linear probe (encoder + head weights) from ``train_linear_probe`` checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    num_classes = len(class_to_idx)
    model = LinearProbe(encoder, embed_dim, num_classes).to(device)
    # strict=False so checkpoints predating feat_bn still load.
    bad = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if bad.missing_keys or bad.unexpected_keys:
        warnings.warn(
            "Linear probe checkpoint loaded with strict=False: "
            f"{len(bad.missing_keys)} missing, {len(bad.unexpected_keys)} unexpected. "
            "(Expected for pre-BN checkpoints — feat_bn will use init running stats.)",
            stacklevel=2,
        )
    model.eval()
    return model, class_to_idx


def make_fetal_planes_probe_dataloader(
    class_to_idx: Dict[str, int],
    dataset_root: str,
    *,
    images_dir: str = "Images",
    csv_file: str = "FETAL_PLANES_DB_data.csv",
    max_image_height: int = 224,
    patch_size: int = 16,
    image_transform: Optional[Callable[[dict], torch.Tensor]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    split: str = "val",
    shuffle: bool = False,
    # Legacy / deprecated:
    image_size: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> DataLoader:
    """
    Build train or val DataLoader for linear probe evaluation (same setup as ``train_linear_probe``).

    Args:
        class_to_idx: Composite label -> class index mapping.
        dataset_root: Root directory of FETAL_PLANES_DB.
        max_image_height: Shrink images taller than this (aspect preserved).
        patch_size: Patch size used for batch padding.
        split: ``"train"`` or ``"val"`` (uses CSV ``Train`` column).
        image_size / target_size: **Deprecated** — ignored.
    """
    _warn_legacy_kwargs(image_size=image_size, target_size=target_size)

    if split not in ("train", "val"):
        raise ValueError('split must be "train" or "val"')
    train_flag = split == "train"

    transform = image_transform or _default_image_transform(max_image_height)
    base = FetalPlanesDBDataset(
        root=dataset_root,
        images_dir=images_dir,
        csv_file=csv_file,
        train=train_flag,
        class_to_idx=class_to_idx,
    )
    ds = _ClassificationDatasetWrapper(base, transform)
    collate_fn = partial(pad_classify_collate, patch_size=int(patch_size))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


@torch.no_grad()
def linear_probe_gather_predictions(
    model: LinearProbe,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred)`` as numpy arrays of shape ``[N]``."""
    model.eval()
    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []
    for images, pad_masks, labels in loader:
        images = images.to(device)
        pad_masks = pad_masks.to(device)
        logits = model(images, pad_mask=pad_masks)
        pred = logits.argmax(dim=1).cpu().numpy()
        y_pred_list.append(pred)
        y_true_list.append(labels.cpu().numpy())
    if not y_true_list:
        return np.array([]), np.array([])
    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


def linear_probe_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix (rows = true class, cols = predicted)."""
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
