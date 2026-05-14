"""
Linear probe classifier: frozen encoder + linear head on fetal_planes_db.

Environment variables (all optional, with defaults):
    DATASET_ROOT              root directory of FETAL_PLANES_DB
    IMAGES_DIR                Images
    ENCODER_TYPE              mae | vicreg | timm  (default: mae)
    PATH_ENCODER_CHECKPOINT   path to .pt checkpoint
    TIMM_MODEL                timm model name (when ENCODER_TYPE=timm)
    EMBED_DIM                 512
    MAX_IMAGE_HEIGHT          224
    EPOCHS                    100
    BATCH_SIZE                64
    LR                        1e-3
    WEIGHT_DECAY              0.0
    NUM_WORKERS               4
    CHECKPOINT_DIR            ./checkpoints/linear_probe
    SAVE_EVERY                10
    DEVICE                    auto
"""

import os
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.mae_classify.model import (
    ClassificationDatasetWrapper,
    LinearProbe,
    default_image_transform,
    pad_classify_collate,
)
from datasets_adapters.fetal_planes_db.fpd_dataset import FetalPlanesDBDataset

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATASET_ROOT             = os.environ.get("DATASET_ROOT", "").strip()
IMAGES_DIR               = os.environ.get("IMAGES_DIR", "Images").strip()
ENCODER_TYPE             = os.environ.get("ENCODER_TYPE", "mae").strip().lower()
PATH_ENCODER_CHECKPOINT  = os.environ.get("PATH_ENCODER_CHECKPOINT", "").strip()
TIMM_MODEL               = os.environ.get("TIMM_MODEL", "").strip()
EMBED_DIM                = int(os.environ.get("EMBED_DIM", 512))
MAX_IMAGE_HEIGHT         = int(os.environ.get("MAX_IMAGE_HEIGHT", 224))
EPOCHS                   = int(os.environ.get("EPOCHS", 100))
BATCH_SIZE               = int(os.environ.get("BATCH_SIZE", 64))
LR                       = float(os.environ.get("LR", 1e-3))
WEIGHT_DECAY             = float(os.environ.get("WEIGHT_DECAY", 0.0))
NUM_WORKERS              = int(os.environ.get("NUM_WORKERS", 4))
CHECKPOINT_DIR           = os.environ.get("CHECKPOINT_DIR", "./checkpoints/linear_probe").strip()
SAVE_EVERY               = int(os.environ.get("SAVE_EVERY", 10))

_device_env = os.environ.get("DEVICE", "").strip()
DEVICE = torch.device(_device_env if _device_env else ("cuda" if torch.cuda.is_available() else "cpu"))


# ─────────────────────────────────────────────────────────────────────────────
# Encoder loading
# ─────────────────────────────────────────────────────────────────────────────

def load_encoder(
    encoder_type: str,
    checkpoint: str,
    timm_model: str,
    device: torch.device,
) -> Tuple[nn.Module, int]:
    """Load encoder and return (encoder, embed_dim)."""
    if encoder_type == "timm" or timm_model:
        from embeddings.timm_encoder import TimmViTEncoder
        name = timm_model or checkpoint
        enc = TimmViTEncoder(name, pretrained=True)
    elif encoder_type == "mae":
        from embeddings.vit.train import load_checkpoint as mae_load
        enc, info = mae_load(checkpoint, device=device)
        print(f"Loaded MAE encoder from epoch {info.get('epoch', '?')}")
    elif encoder_type == "vicreg":
        from embeddings.vicreg.train import load_checkpoint as vicreg_load
        model, info = vicreg_load(checkpoint, device=device)
        enc = model.encoder
        print(f"Loaded VICReg encoder from epoch {info.get('epoch', '?')}")
    else:
        raise ValueError(f"Unknown ENCODER_TYPE: {encoder_type!r}")

    enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    embed_dim = getattr(enc, "embed_dim", EMBED_DIM)
    return enc, embed_dim


# ─────────────────────────────────────────────────────────────────────────────
# Class mapping helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_class_mapping(dataset: FetalPlanesDBDataset) -> Dict[str, int]:
    composites = [label["composite"] for label in dataset.labels]
    return {cls: idx for idx, cls in enumerate(sorted(set(composites)))}


def _infer_patch_size(encoder: nn.Module, default: int = 16) -> int:
    ps = getattr(encoder, "patch_size", None)
    if ps is None:
        ps = getattr(getattr(encoder, "patch_embed", None), "patch_size", None)
    try:
        return int(ps) if ps is not None else default
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_linear_probe(
    encoder: nn.Module,
    embed_dim: int,
    dataset_root: str,
    *,
    images_dir: str = "Images",
    csv_file: str = "FETAL_PLANES_DB_data.csv",
    max_image_height: int = 224,
    patch_size: Optional[int] = None,
    image_transform: Optional[Callable] = None,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[str] = None,
    save_every: int = 10,
    show_plot: bool = False,
) -> Dict[str, List[float]]:
    """
    Train a linear probe (frozen encoder + linear head) on fetal_planes_db.

    Returns history dict with train/val loss, accuracy, and balanced accuracy.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if patch_size is None:
        patch_size = _infer_patch_size(encoder)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    transform = image_transform or default_image_transform(max_image_height)
    collate_fn = partial(pad_classify_collate, patch_size=int(patch_size))

    scout = FetalPlanesDBDataset(root=dataset_root, images_dir=images_dir, csv_file=csv_file, train=True)
    class_to_idx = _get_class_mapping(scout)
    num_classes = len(class_to_idx)

    train_loader = DataLoader(
        ClassificationDatasetWrapper(
            FetalPlanesDBDataset(root=dataset_root, images_dir=images_dir, csv_file=csv_file,
                                 train=True, class_to_idx=class_to_idx),
            transform,
        ),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ClassificationDatasetWrapper(
            FetalPlanesDBDataset(root=dataset_root, images_dir=images_dir, csv_file=csv_file,
                                 train=False, class_to_idx=class_to_idx),
            transform,
        ),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=collate_fn,
    )

    model = LinearProbe(encoder, embed_dim, num_classes).to(device)
    optimizer = optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "train_bal_acc": [],
        "val_loss":   [], "val_acc":   [], "val_bal_acc":   [],
    }

    for epoch in range(epochs):
        # --- train ---
        model.train()
        tr_loss = tr_correct = tr_total = 0
        tr_labels: List[torch.Tensor] = []
        tr_preds:  List[torch.Tensor] = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False)
        for images, pad_masks, labels in pbar:
            images, pad_masks, labels = images.to(device), pad_masks.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images, pad_mask=pad_masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            pred = logits.argmax(1)
            tr_loss += loss.item()
            tr_correct += (pred == labels).sum().item()
            tr_total += labels.size(0)
            tr_labels.append(labels.cpu())
            tr_preds.append(pred.detach().cpu())
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*tr_correct/tr_total:.2f}%")

        tr_loss /= len(train_loader)
        tr_acc = tr_correct / tr_total if tr_total else 0.0
        tr_bal = _balanced_accuracy(torch.cat(tr_labels).numpy(), torch.cat(tr_preds).numpy())

        # --- val ---
        model.eval()
        vl_loss = vl_correct = vl_total = 0
        vl_labels: List[torch.Tensor] = []
        vl_preds:  List[torch.Tensor] = []
        with torch.no_grad():
            for images, pad_masks, labels in val_loader:
                images, pad_masks, labels = images.to(device), pad_masks.to(device), labels.to(device)
                logits = model(images, pad_mask=pad_masks)
                loss = criterion(logits, labels)
                pred = logits.argmax(1)
                vl_loss += loss.item()
                vl_correct += (pred == labels).sum().item()
                vl_total += labels.size(0)
                vl_labels.append(labels.cpu())
                vl_preds.append(pred.cpu())

        vl_loss /= len(val_loader) if val_loader else 1
        vl_acc = vl_correct / vl_total if vl_total else 0.0
        vl_bal = _balanced_accuracy(torch.cat(vl_labels).numpy(), torch.cat(vl_preds).numpy())

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_bal_acc"].append(tr_bal)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_bal_acc"].append(vl_bal)

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"train_loss={tr_loss:.4f} train_acc={100*tr_acc:.2f}% train_bal_acc={100*tr_bal:.2f}% | "
            f"val_loss={vl_loss:.4f} val_acc={100*vl_acc:.2f}% val_bal_acc={100*vl_bal:.2f}%"
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
                "max_image_height": max_image_height,
                "patch_size": patch_size,
                "history": history,
            }, path)
            print(f"  → checkpoint saved: {path}")

    if checkpoint_dir or show_plot:
        plot_path = os.path.join(checkpoint_dir, "linear_probe_curves.png") if checkpoint_dir else None
        plot_linear_probe_history(history, save_path=plot_path, show=show_plot)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_linear_probe_from_checkpoint(
    checkpoint_path: str,
    encoder: nn.Module,
    embed_dim: int,
    device: torch.device,
) -> Tuple[LinearProbe, Dict[str, int]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    model = LinearProbe(encoder, embed_dim, len(class_to_idx)).to(device)
    bad = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if bad.missing_keys or bad.unexpected_keys:
        warnings.warn(
            f"Checkpoint loaded with strict=False: {len(bad.missing_keys)} missing, "
            f"{len(bad.unexpected_keys)} unexpected.",
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
    image_transform: Optional[Callable] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    split: str = "val",
    shuffle: bool = False,
) -> DataLoader:
    if split not in ("train", "val"):
        raise ValueError('split must be "train" or "val"')
    transform = image_transform or default_image_transform(max_image_height)
    ds = ClassificationDatasetWrapper(
        FetalPlanesDBDataset(
            root=dataset_root, images_dir=images_dir, csv_file=csv_file,
            train=(split == "train"), class_to_idx=class_to_idx,
        ),
        transform,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, collate_fn=partial(pad_classify_collate, patch_size=patch_size),
    )


@torch.no_grad()
def gather_predictions(
    model: LinearProbe, loader: DataLoader, device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred)`` numpy arrays of shape ``[N]``."""
    model.eval()
    y_true, y_pred = [], []
    for images, pad_masks, labels in loader:
        logits = model(images.to(device), pad_mask=pad_masks.to(device))
        y_pred.append(logits.argmax(1).cpu().numpy())
        y_true.append(labels.numpy())
    if not y_true:
        return np.array([]), np.array([])
    return np.concatenate(y_true), np.concatenate(y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    from sklearn.metrics import confusion_matrix as sk_cm
    return sk_cm(y_true, y_pred, labels=np.arange(num_classes))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_linear_probe_history(
    history: Dict[str, List[float]],
    *,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    n = len(history["train_loss"])
    epochs = np.arange(1, n + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Cross-entropy loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [100*x for x in history["train_acc"]], label="train")
    axes[1].plot(epochs, [100*x for x in history["val_acc"]], label="val")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, [100*x for x in history["train_bal_acc"]], label="train (bal.)")
    axes[2].plot(epochs, [100*x for x in history["val_bal_acc"]], label="val (bal.)")
    axes[2].set(xlabel="Epoch", ylabel="Balanced accuracy (%)", title="Balanced accuracy")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not DATASET_ROOT:
        raise ValueError("DATASET_ROOT must be set.")
    if ENCODER_TYPE not in ("mae", "vicreg", "timm"):
        raise ValueError(f"ENCODER_TYPE must be mae|vicreg|timm, got {ENCODER_TYPE!r}")
    if ENCODER_TYPE in ("mae", "vicreg") and not PATH_ENCODER_CHECKPOINT:
        raise ValueError(f"PATH_ENCODER_CHECKPOINT required for ENCODER_TYPE={ENCODER_TYPE!r}")
    if ENCODER_TYPE == "timm" and not TIMM_MODEL:
        raise ValueError("TIMM_MODEL must be set when ENCODER_TYPE='timm'")

    encoder, embed_dim = load_encoder(ENCODER_TYPE, PATH_ENCODER_CHECKPOINT, TIMM_MODEL, DEVICE)

    train_linear_probe(
        encoder=encoder,
        embed_dim=embed_dim,
        dataset_root=DATASET_ROOT,
        images_dir=IMAGES_DIR,
        max_image_height=MAX_IMAGE_HEIGHT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        checkpoint_dir=CHECKPOINT_DIR,
        save_every=SAVE_EVERY,
        show_plot=False,
    )


if __name__ == "__main__":
    main()
