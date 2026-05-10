"""
Classification baseline using a frozen VICReg encoder with a trainable MLP head.

Loads a pretrained VICReg checkpoint, freezes the encoder, trains a
classification head on labels from ``processed_iter_1.csv``, and reports
balanced accuracy, F1, and a confusion matrix.

``VICRegModel.encode()`` is used for embedding extraction (no projector —
raw encoder representations, same as what RankME measures during SSL training).

All paths and hyperparameters are configurable via environment variables
(identical set to ``run.py``).

Usage::

    DIR_IMAGES=/data/npy \
    PATH_MAE_CHECKPOINT=checkpoints/vicreg_final.pt \
    python baselines/mae_classify/run_vicreg.py
"""

import json
import math
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration (environment variables with defaults)
# ---------------------------------------------------------------------------

DIR_IMAGES = os.environ.get("DIR_IMAGES", "<path_to_images>")
PATH_CSV = os.environ.get("PATH_CSV", "../processed_iter_1.csv")
PATH_MAE_CHECKPOINT = os.environ.get("PATH_MAE_CHECKPOINT", "")
IMAGE_EXT = os.environ.get("IMAGE_EXT", "")

EXCLUDE_CLASS_IDS = os.environ.get("EXCLUDE_CLASS_IDS", "4")
EPOCHS = int(os.environ.get("EPOCHS", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
LR = float(os.environ.get("LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", "0.2"))
HEAD_HIDDEN_DIM = int(os.environ.get("HEAD_HIDDEN_DIM", "0"))
HEAD_DROPOUT = float(os.environ.get("HEAD_DROPOUT", "0.1"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
MAX_IMAGE_HEIGHT = int(
    os.environ.get(
        "MAX_IMAGE_HEIGHT",
        os.environ.get("IMAGE_HEIGHT", os.environ.get("IMAGE_SIZE", "224")),
    )
)
STANDARDIZE_INPUT = os.environ.get("STANDARDIZE_INPUT", "1").strip().lower() in (
    "1", "true", "yes",
)
SEED = int(os.environ.get("SEED", "42"))
CONFIG_PATH = os.environ.get("CONFIG_PATH", "run_config.json")
PATH_HEAD_CHECKPOINT = os.environ.get("PATH_HEAD_CHECKPOINT", "best_head.pt")
TIMM_MODEL = os.environ.get("TIMM_MODEL", "").strip()
METRICS_CSV = os.environ.get("METRICS_CSV", "training_metrics.csv")
_mode_raw = os.environ.get("MODE", "train").strip().lower()
_eval_flag = os.environ.get("EVAL_ONLY", "").strip().lower() in ("1", "true", "yes")
MODE = "eval" if _eval_flag else _mode_raw
DEVICE = os.environ.get("DEVICE", "").strip()
RANDOM_ENCODER = os.environ.get("RANDOM_ENCODER", "0").strip().lower() in ("1", "true", "yes")

if DEVICE:
    device = torch.device(DEVICE)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_exclude_ids(raw: str) -> Set[int]:
    if not raw.strip():
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _standardize_per_image(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = img.mean()
    std = img.std().clamp_min(eps)
    return (img - mean) / std


def load_npy_as_grayscale_tensor(
    path: str, max_image_height: int, standardize: bool = True,
) -> torch.Tensor:
    import torchvision.transforms.functional as F

    npy = np.load(path)
    if npy.dtype != np.uint8:
        npy = np.clip(npy, 0, 255).astype(np.uint8)

    img = Image.fromarray(npy, mode="RGB").convert("L")
    tensor = torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0

    _, h, w = tensor.shape
    if max_image_height and h > max_image_height:
        new_h = int(max_image_height)
        new_w = max(1, int(round(w * new_h / h)))
        tensor = F.resize(
            tensor, [new_h, new_w],
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
        )

    if standardize:
        tensor = _standardize_per_image(tensor)
    return tensor


def pad_classify_collate(
    batch: List[Tuple[torch.Tensor, int, str]], patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    imgs = [it[0] for it in batch]
    labels = torch.tensor([int(it[1]) for it in batch], dtype=torch.long)
    paths = [str(it[2]) for it in batch]
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
    return images, pad_masks, labels, paths


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClassificationDataset(Dataset):
    def __init__(
        self,
        dir_images: str,
        path_csv: str,
        max_image_height: int = 224,
        exclude_class_ids: Optional[Set[int]] = None,
        image_ext: str = "",
        standardize: bool = True,
    ):
        if exclude_class_ids is None:
            exclude_class_ids = set()

        df = pd.read_csv(path_csv)
        df = df[~df["class_idx"].isin(exclude_class_ids)].reset_index(drop=True)

        sorted_ids = sorted(df["class_idx"].unique())
        self.orig_to_seq = {orig: seq for seq, orig in enumerate(sorted_ids)}
        self.seq_to_orig = {seq: orig for orig, seq in self.orig_to_seq.items()}
        self.num_classes = len(sorted_ids)

        self.class_names: Dict[int, str] = {}
        for orig_id in sorted_ids:
            names = df.loc[df["class_idx"] == orig_id, "class"].unique()
            self.class_names[self.orig_to_seq[orig_id]] = names[0]

        self.data: List[Dict] = []
        for _, row in df.iterrows():
            self.data.append({
                "img": os.path.join(dir_images, f"{row['image']}{image_ext}.npy"),
                "class_name": row["class"],
                "orig_idx": int(row["class_idx"]),
                "seq_idx": self.orig_to_seq[int(row["class_idx"])],
            })

        self.max_image_height = max_image_height
        self.standardize = bool(standardize)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        entry = self.data[index]
        img = load_npy_as_grayscale_tensor(
            entry["img"], self.max_image_height, standardize=self.standardize,
        )
        return img, entry["seq_idx"], entry["img"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        hid = hidden_dim if hidden_dim > 0 else embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VICRegClassifier(nn.Module):
    """Frozen VICReg encoder + trainable classification head.

    Uses ``VICRegModel.encode()`` which returns raw encoder embeddings
    (before the projector), the same representations evaluated by RankME.
    """

    def __init__(self, vicreg_model: nn.Module, head: nn.Module):
        super().__init__()
        self.vicreg = vicreg_model
        self.head = head

        self.vicreg.eval()
        for p in self.vicreg.parameters():
            p.requires_grad = False

    def forward(
        self, imgs: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            emb = self.vicreg.encode(imgs, pad_mask=pad_mask)
        return self.head(emb)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: VICRegClassifier,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device_: torch.device,
    epoch: int,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for imgs, pad_masks, labels, _paths in pbar:
        imgs = imgs.to(device_)
        pad_masks = pad_masks.to(device_)
        labels = labels.to(device_)

        optimizer.zero_grad()
        logits = model(imgs, pad_mask=pad_masks)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        all_true.append(labels.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / total:.1f}%")

    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    train_bal = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    return total_loss / max(total, 1), correct / max(total, 1), train_bal


@torch.no_grad()
def evaluate(
    model: VICRegClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device_: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    for imgs, pad_masks, labels, _paths in loader:
        imgs = imgs.to(device_)
        pad_masks = pad_masks.to(device_)
        labels = labels.to(device_)

        logits = model(imgs, pad_mask=pad_masks)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        all_true.append(labels.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    return total_loss / max(total, 1), correct / max(total, 1), y_true, y_pred


# ---------------------------------------------------------------------------
# Stratified train/val split
# ---------------------------------------------------------------------------

def stratified_split(
    dataset: ClassificationDataset,
    val_fraction: float,
    seed: int,
) -> Tuple[Subset, Subset]:
    rng = np.random.RandomState(seed)
    labels = np.array([d["seq_idx"] for d in dataset.data])
    train_indices: List[int] = []
    val_indices: List[int] = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * val_fraction))
        val_indices.extend(cls_idx[:n_val].tolist())
        train_indices.extend(cls_idx[n_val:].tolist())

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# ---------------------------------------------------------------------------
# Metrics & output
# ---------------------------------------------------------------------------

def write_training_metrics_csv(history: Dict[str, List[float]], path: str) -> None:
    n = len(history["train_loss"])
    if n == 0:
        return

    rows: List[Dict[str, object]] = []
    for i in range(n):
        rows.append({
            "epoch": i + 1,
            "train_loss": float(history["train_loss"][i]),
            "train_accuracy": float(history["train_acc"][i]),
            "train_balanced_accuracy": float(history["train_bal_acc"][i]),
            "val_loss": float(history["val_loss"][i]),
            "val_accuracy": float(history["val_acc"][i]),
            "val_balanced_accuracy": float(history["val_bal_acc"][i]),
        })

    out = os.path.abspath(path)
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def compute_and_print_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Dict[int, str],
    num_classes: int,
) -> Dict:
    labels = list(range(num_classes))
    target_names = [class_names.get(i, str(i)) for i in labels]

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\nBalanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1:          {f1_macro:.4f}\n")
    print("Classification report:")
    print(classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0,
    ))
    print("Confusion matrix:")
    print(cm)

    return {
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_per_class": {name: float(f1_per_class[i]) for i, name in enumerate(target_names)},
        "confusion_matrix": cm,
        "class_names": target_names,
    }


def build_train_config(
    encoder_checkpoint: str,
    encoder_info: Dict,
    dataset: ClassificationDataset,
    n_train: int,
    n_val: int,
) -> Dict:
    model_cfg = encoder_info.get("model_config", {})
    head_hid = HEAD_HIDDEN_DIM if HEAD_HIDDEN_DIM > 0 else model_cfg.get("embed_dim", "?")
    return {
        "vicreg_checkpoint": os.path.abspath(encoder_checkpoint),
        "vicreg_epoch": encoder_info.get("epoch"),
        "vicreg_model_config": model_cfg,
        "random_encoder": RANDOM_ENCODER,
        "dir_images": os.path.abspath(DIR_IMAGES),
        "path_csv": os.path.abspath(PATH_CSV),
        "image_ext": IMAGE_EXT,
        "exclude_class_ids": sorted(_parse_exclude_ids(EXCLUDE_CLASS_IDS)),
        "max_image_height": MAX_IMAGE_HEIGHT,
        "standardize_input": STANDARDIZE_INPUT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "val_split": VAL_SPLIT,
        "head_hidden_dim": head_hid,
        "head_dropout": HEAD_DROPOUT,
        "num_workers": NUM_WORKERS,
        "seed": SEED,
        "device": str(device),
        "num_classes": dataset.num_classes,
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "metrics_csv": os.path.abspath(METRICS_CSV),
        "class_mapping": {
            str(seq): {"orig_idx": dataset.seq_to_orig[seq], "name": dataset.class_names[seq]}
            for seq in sorted(dataset.class_names)
        },
    }


def save_run_config(
    train_config: Dict,
    history: Dict[str, List[float]],
    final_metrics: Optional[Dict] = None,
    path: str = "run_config.json",
):
    payload = {"train_config": train_config, "history": history}
    if final_metrics is not None:
        serialisable_metrics = {k: v for k, v in final_metrics.items() if k != "confusion_matrix"}
        serialisable_metrics["confusion_matrix"] = final_metrics["confusion_matrix"].tolist()
        payload["final_metrics"] = serialisable_metrics

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved {path}")


def save_results(
    metrics: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    paths: List[str],
    class_names: Dict[int, str],
):
    target_names = metrics["class_names"]

    row = {
        "model": "VICReg-ViT",
        "balanced_accuracy": metrics["balanced_accuracy"],
        "f1_macro": metrics["f1_macro"],
    }
    for name, val in metrics["f1_per_class"].items():
        row[f"f1_{name}"] = val

    pd.DataFrame([row]).to_csv("test_results.csv", index=False)
    print("Saved test_results.csv")

    cm_df = pd.DataFrame(metrics["confusion_matrix"], index=target_names, columns=target_names)
    cm_df.to_csv("confusion_matrix.csv")
    print("Saved confusion_matrix.csv")

    pred_names = [class_names.get(int(p), str(p)) for p in y_pred]
    true_names = [class_names.get(int(t), str(t)) for t in y_true]
    with open("test_prediction.json", "w", encoding="utf-8") as f:
        json.dump(
            {"prediction": pred_names, "label": true_names, "paths": paths},
            f, indent=2, ensure_ascii=False,
        )
    print("Saved test_prediction.json")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_head_checkpoint(model, optimizer, epoch, val_bal_acc, train_config, history, path):
    torch.save({
        "head_state_dict": model.head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_bal_acc": val_bal_acc,
        "train_config": train_config,
        "history": history,
    }, path)


def _load_head_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.head.load_state_dict(ckpt["head_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _final_evaluate(model, val_loader, val_ds, ds, criterion):
    val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    val_paths = [ds.data[idx]["img"] for idx in val_ds.indices]
    metrics = compute_and_print_metrics(y_true, y_pred, ds.class_names, ds.num_classes)
    save_results(metrics, y_true, y_pred, val_paths, ds.class_names)
    return metrics


def main():
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(script_dir, "..", ".."))
    sys.path.insert(0, script_dir)
    from embeddings.vicreg.train import load_checkpoint

    if not PATH_MAE_CHECKPOINT and not TIMM_MODEL:
        raise RuntimeError(
            "Set PATH_MAE_CHECKPOINT (VICReg checkpoint) or TIMM_MODEL (timm model name)"
        )
    if MODE not in ("train", "resume", "eval"):
        raise RuntimeError(f"MODE must be train, resume, or eval -- got: {MODE}")

    print(f"Device: {device}")
    print(f"Mode:   {MODE}")
    print(f"Images dir: {DIR_IMAGES}")
    print(f"CSV: {PATH_CSV}")

    exclude_ids = _parse_exclude_ids(EXCLUDE_CLASS_IDS)
    print(f"Excluding class_idx: {exclude_ids}")

    if TIMM_MODEL:
        from embeddings.timm_encoder import TimmViTEncoder
        print(f"Timm model: {TIMM_MODEL}")
        encoder = TimmViTEncoder(TIMM_MODEL, pretrained=not RANDOM_ENCODER).to(device)
        if RANDOM_ENCODER:
            encoder._init_weights()
            print("*** RANDOM ENCODER baseline — timm architecture, weights re-initialised ***")
        vicreg_model = encoder
        ckpt_info = {"epoch": "pretrained", "model_config": {"timm_model": TIMM_MODEL}}
    else:
        print(f"Checkpoint: {PATH_MAE_CHECKPOINT}")
        vicreg_model, ckpt_info = load_checkpoint(PATH_MAE_CHECKPOINT, device=device)
        if RANDOM_ENCODER:
            vicreg_model.encoder._init_weights()
            print("*** RANDOM ENCODER baseline — architecture from checkpoint, weights re-initialised ***")

    embed_dim = vicreg_model.embed_dim if TIMM_MODEL else vicreg_model.encoder.embed_dim
    patch_size = vicreg_model.patch_size if TIMM_MODEL else vicreg_model.encoder.patch_size
    print(
        f"Encoder embed_dim={embed_dim}, patch_size={patch_size}, "
        f"loaded from {TIMM_MODEL or ckpt_info.get('epoch', '?')}"
    )

    ds = ClassificationDataset(
        dir_images=DIR_IMAGES,
        path_csv=PATH_CSV,
        max_image_height=MAX_IMAGE_HEIGHT,
        exclude_class_ids=exclude_ids,
        image_ext=IMAGE_EXT,
        standardize=STANDARDIZE_INPUT,
    )
    print(
        f"Input standardisation: {'ON' if STANDARDIZE_INPUT else 'OFF'} "
        f"(must match the VICReg pre-training transform)"
    )
    num_classes = ds.num_classes
    print(f"Dataset: {len(ds)} samples, {num_classes} classes")
    for seq_idx in sorted(ds.class_names):
        count = sum(1 for d in ds.data if d["seq_idx"] == seq_idx)
        print(f"  [{seq_idx}] {ds.class_names[seq_idx]}: {count}")

    train_ds, val_ds = stratified_split(ds, VAL_SPLIT, SEED)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    def _collate(batch):
        return pad_classify_collate(batch, patch_size)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False, drop_last=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
        collate_fn=_collate,
    )

    criterion = nn.CrossEntropyLoss()

    # ===================================================================
    # MODE: eval
    # ===================================================================
    if MODE == "eval":
        print(f"\n*** Evaluation-only mode -- loading head from {PATH_HEAD_CHECKPOINT} ***")
        head_ckpt = torch.load(PATH_HEAD_CHECKPOINT, map_location=device, weights_only=False)
        saved_cfg = head_ckpt.get("train_config", {})
        head = MLPHead(embed_dim, num_classes,
                       saved_cfg.get("head_hidden_dim", HEAD_HIDDEN_DIM),
                       saved_cfg.get("head_dropout", HEAD_DROPOUT))
        model = VICRegClassifier(vicreg_model, head).to(device)
        model.head.load_state_dict(head_ckpt["head_state_dict"])
        print(f"Head loaded (trained epoch {head_ckpt.get('epoch', '?')}, "
              f"val_bal_acc={100 * head_ckpt.get('val_bal_acc', 0):.2f}%)")

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        _final_evaluate(model, val_loader, val_ds, ds, criterion)
        return

    # ===================================================================
    # MODE: train or resume
    # ===================================================================
    start_epoch = 1
    best_val_bal_acc = 0.0
    best_epoch = 0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "train_bal_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_bal_acc": [],
        "val_f1": [],
    }

    if MODE == "resume":
        print(f"\n*** Resume mode -- loading head from {PATH_HEAD_CHECKPOINT} ***")
        head_ckpt = torch.load(PATH_HEAD_CHECKPOINT, map_location=device, weights_only=False)
        saved_cfg = head_ckpt.get("train_config", {})
        head = MLPHead(embed_dim, num_classes,
                       saved_cfg.get("head_hidden_dim", HEAD_HIDDEN_DIM),
                       saved_cfg.get("head_dropout", HEAD_DROPOUT))
        model = VICRegClassifier(vicreg_model, head).to(device)
        model.head.load_state_dict(head_ckpt["head_state_dict"])
        optimizer = optim.AdamW(model.head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        if "optimizer_state_dict" in head_ckpt:
            optimizer.load_state_dict(head_ckpt["optimizer_state_dict"])
        saved_epoch = head_ckpt.get("epoch", 0)
        start_epoch = saved_epoch + 1
        best_val_bal_acc = head_ckpt.get("val_bal_acc", 0.0)
        best_epoch = saved_epoch
        if "history" in head_ckpt:
            saved = head_ckpt["history"]
            n_ep = len(saved.get("train_loss", []))
            for k in history:
                v = list(saved.get(k, []))
                while len(v) < n_ep:
                    v.append(float("nan"))
                history[k] = v[:n_ep]
        print(f"Resumed from epoch {saved_epoch} "
              f"(val_bal_acc={100 * best_val_bal_acc:.2f}%), "
              f"will train to epoch {EPOCHS}")
    else:
        head = MLPHead(embed_dim, num_classes, HEAD_HIDDEN_DIM, HEAD_DROPOUT)
        model = VICRegClassifier(vicreg_model, head).to(device)
        optimizer = optim.AdamW(model.head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Parameters: {n_trainable:,} trainable | {n_frozen:,} frozen")

    train_config = build_train_config(
        TIMM_MODEL or PATH_MAE_CHECKPOINT, ckpt_info, ds, len(train_ds), len(val_ds),
    )
    print(f"Per-epoch metrics: {os.path.abspath(METRICS_CSV)}")

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss, train_acc, train_bal_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
        )
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        val_bal_acc = balanced_accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
        val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) > 0 else 0.0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_bal_acc"].append(train_bal_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_bal_acc"].append(val_bal_acc)
        history["val_f1"].append(val_f1)

        write_training_metrics_csv(history, METRICS_CSV)

        improved = val_bal_acc > best_val_bal_acc
        if improved:
            best_val_bal_acc = val_bal_acc
            best_epoch = epoch
            _save_head_checkpoint(
                model, optimizer, epoch, val_bal_acc,
                train_config, history, PATH_HEAD_CHECKPOINT,
            )

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={100 * train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={100 * val_acc:.2f}% "
            f"val_bal_acc={100 * val_bal_acc:.2f}% val_f1={val_f1:.4f}"
            + ("  *best*" if improved else "")
        )

    print(f"\nBest validation balanced accuracy: {100 * best_val_bal_acc:.2f}% at epoch {best_epoch}")

    best_ckpt = torch.load(PATH_HEAD_CHECKPOINT, map_location=device, weights_only=False)
    model.head.load_state_dict(best_ckpt["head_state_dict"])

    print("\n" + "=" * 60)
    print("FINAL RESULTS (best checkpoint on validation set)")
    print("=" * 60)

    metrics = _final_evaluate(model, val_loader, val_ds, ds, criterion)
    save_run_config(train_config, history, final_metrics=metrics, path=CONFIG_PATH)


if __name__ == "__main__":
    main()
