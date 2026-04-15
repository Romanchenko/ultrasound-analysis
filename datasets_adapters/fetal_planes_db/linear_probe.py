"""
Linear probe evaluation: train a classification head on frozen encoder embeddings.

Evaluates any pretrained encoder on the fetal_planes_db (Plane, Brain_plane) classification
task. The encoder is frozen; only a linear head is trained.

Usage::

    from datasets_adapters.fetal_planes_db.linear_probe import train_linear_probe

    # With MAE encoder
    from embeddings.vit.train import load_checkpoint
    encoder, _ = load_checkpoint("checkpoints/mae_final.pt")

    # With any encoder that has .encode(imgs) -> [B, embed_dim]
    history = train_linear_probe(
        encoder=encoder,
        embed_dim=encoder.embed_dim,  # or 384 for MAE default
        dataset_root="path/to/FETAL_PLANES_DB",
        image_size=224,
        epochs=50,
    )
"""

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
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
      - .encode(imgs) -> [B, embed_dim], or
      - .forward(imgs) -> [B, embed_dim] (embedding as direct output)

    The encoder is kept in eval mode and its parameters are frozen.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)
        self._use_encode = hasattr(encoder, "encode")

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_encode:
            return self.encoder.encode(x)
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            emb = self._get_embeddings(x)
        return self.head(emb)


# =====================================================================
# Dataset wrapper & transform
# =====================================================================

def _default_image_transform(image_size: int):
    """Build transform: resize to image_size×image_size, ensure [1,H,W] grayscale."""

    def transform(sample: dict) -> torch.Tensor:
        img = sample["image"]
        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)
        if img.dim() == 2:
            img = img.unsqueeze(0)
        if img.size(0) > 1:
            img = img.mean(dim=0, keepdim=True)
        img = T.functional.resize(
            img,
            [image_size, image_size],
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )
        return img

    return transform


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
    image_size: int = 224,
    target_size: Optional[Tuple[int, int]] = None,
    image_transform: Optional[Callable[[dict], torch.Tensor]] = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[str] = None,
    save_every: int = 10,
    show_plot: bool = False,
) -> Dict[str, List[float]]:
    """
    Train a linear classification head on frozen encoder embeddings for (Plane, Brain_plane).

    Args:
        encoder: Pretrained encoder (any module with .encode(x) or callable returning
                 [B, embed_dim]). Will be frozen.
        embed_dim: Output dimension of the encoder embeddings.
        dataset_root: Root directory of FETAL_PLANES_DB.
        images_dir: Name of images subdirectory.
        csv_file: Name of CSV metadata file.
        image_size: Spatial size to resize images to (must match encoder expectation).
        target_size: (H, W) for dataset's initial load. Defaults to (image_size, image_size).
        image_transform: Optional callable taking a sample dict and returning an image
                         tensor [1, H, W]. If None, uses default resize to image_size.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate for the linear head.
        weight_decay: Weight decay.
        num_workers: DataLoader workers.
        device: Device (auto-detect if None).
        checkpoint_dir: Where to save checkpoints (None = no saving).
        save_every: Save checkpoint every N epochs.
        show_plot: If True, display the loss/accuracy/balanced-accuracy figure (e.g. notebooks).

    Returns:
        history: Dict with ``train_loss``, ``train_acc``, ``train_bal_acc``, ``val_loss``,
                 ``val_acc``, ``val_bal_acc`` lists (accuracies as fractions in ``[0, 1]``).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if target_size is None:
        target_size = (image_size, image_size)

    # Ensure encoder is frozen and in eval mode
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    transform = image_transform or _default_image_transform(image_size)

    # Scout dataset to build class mapping
    scout = FetalPlanesDBDataset(
        root=dataset_root,
        images_dir=images_dir,
        csv_file=csv_file,
        target_size=target_size,
        transform=[T.ToTensor()],
        train=True,
    )
    class_to_idx = _get_class_mapping(scout)
    num_classes = len(class_to_idx)

    # Classification datasets with proper splits
    train_base = FetalPlanesDBDataset(
        root=dataset_root,
        images_dir=images_dir,
        csv_file=csv_file,
        target_size=target_size,
        transform=[T.ToTensor()],
        train=True,
        class_to_idx=class_to_idx,
    )
    val_base = FetalPlanesDBDataset(
        root=dataset_root,
        images_dir=images_dir,
        csv_file=csv_file,
        target_size=target_size,
        transform=[T.ToTensor()],
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
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = LinearProbe(encoder, embed_dim, num_classes).to(device)
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
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_labels_chunks: List[torch.Tensor] = []
        train_preds_chunks: List[torch.Tensor] = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False)

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
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

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels_chunks: List[torch.Tensor] = []
        val_preds_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
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
                "history": history,
            }, path)
            print(f"  → checkpoint saved: {path}")

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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_to_idx


def make_fetal_planes_probe_dataloader(
    class_to_idx: Dict[str, int],
    dataset_root: str,
    *,
    images_dir: str = "Images",
    csv_file: str = "FETAL_PLANES_DB_data.csv",
    image_size: int = 224,
    target_size: Optional[Tuple[int, int]] = None,
    image_transform: Optional[Callable[[dict], torch.Tensor]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    split: str = "val",
    shuffle: bool = False,
) -> DataLoader:
    """
    Build train or val DataLoader for linear probe evaluation (same setup as ``train_linear_probe``).

    Args:
        split: ``\"train\"`` or ``\"val\"`` (uses CSV ``Train`` column).
    """
    if target_size is None:
        target_size = (image_size, image_size)
    if split not in ("train", "val"):
        raise ValueError('split must be "train" or "val"')
    train_flag = split == "train"

    transform = image_transform or _default_image_transform(image_size)
    base = FetalPlanesDBDataset(
        root=dataset_root,
        images_dir=images_dir,
        csv_file=csv_file,
        target_size=target_size,
        transform=[T.ToTensor()],
        train=train_flag,
        class_to_idx=class_to_idx,
    )
    ds = _ClassificationDatasetWrapper(base, transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
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
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
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
