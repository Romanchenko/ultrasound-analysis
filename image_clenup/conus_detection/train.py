"""
Training script for ultrasound conus segmentation (U-Net + pretrained backbone).

All configuration via environment variables:

    DATASET_ROOT       root directory with XML files and image folders
    BACKBONE           timm backbone name  (default: efficientnet_b0)
    DECODER_CHANNELS   comma-separated decoder widths  (default: 256,128,64,32)
    PRETRAINED         1 = load pretrained backbone weights  (default: 1)
    EPOCHS             default 50
    BATCH_SIZE         default 8
    LR                 default 1e-3
    WEIGHT_DECAY       default 1e-4
    VAL_SPLIT          default 0.2
    NUM_WORKERS        default 4
    TARGET_SIZE        image resize (H=W square)  (default: 512)
    SEED               default 42
    MODE               train | resume | eval  (default: train)
    DEVICE             auto
    PATH_CHECKPOINT    best_conus.pt
    METRICS_CSV        conus_metrics.csv
    CONFIG_PATH        conus_run_config.json
"""

import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from image_clenup.conus_detection.dataset import ConusDataset
from image_clenup.conus_detection.model import ConusUNet

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATASET_ROOT      = os.environ.get("DATASET_ROOT", "")
BACKBONE          = os.environ.get("BACKBONE", "efficientnet_b0").strip()
DECODER_CHANNELS  = [int(x) for x in os.environ.get("DECODER_CHANNELS", "256,128,64,32").split(",")]
PRETRAINED        = int(os.environ.get("PRETRAINED", 1))

EPOCHS       = int(os.environ.get("EPOCHS", 50))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", 8))
LR           = float(os.environ.get("LR", 1e-3))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-4))
VAL_SPLIT    = float(os.environ.get("VAL_SPLIT", 0.2))
NUM_WORKERS  = int(os.environ.get("NUM_WORKERS", 4))
TARGET_SIZE  = int(os.environ.get("TARGET_SIZE", 512))
SEED         = int(os.environ.get("SEED", 42))
MODE         = os.environ.get("MODE", "train").strip().lower()

PATH_CHECKPOINT = os.environ.get("PATH_CHECKPOINT", "best_conus.pt")
METRICS_CSV     = os.environ.get("METRICS_CSV", "conus_metrics.csv")
CONFIG_PATH     = os.environ.get("CONFIG_PATH",  "conus_run_config.json")

_dev = os.environ.get("DEVICE", "").strip()
DEVICE = torch.device(_dev if _dev else ("cuda" if torch.cuda.is_available() else "cpu"))


# ─────────────────────────────────────────────────────────────────────────────
# Loss & metrics
# ─────────────────────────────────────────────────────────────────────────────

def seg_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, float, float]:
    """Combined BCE + Dice loss. Returns (total, bce_val, dice_val)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)

    prob  = torch.sigmoid(logits)
    eps   = 1e-6
    inter = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice  = 1.0 - ((2.0 * inter + eps) / (union + eps)).mean()

    return bce + dice, bce.item(), dice.item()


def compute_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """IoU and Dice for binary segmentation."""
    pred  = (torch.sigmoid(logits) > threshold).float()
    inter = (pred * target).sum().item()
    union = (pred + target).clamp_max(1).sum().item()
    p_sum = pred.sum().item()
    t_sum = target.sum().item()
    iou  = inter / (union  + 1e-6)
    dice = (2 * inter) / (p_sum + t_sum + 1e-6)
    return {"iou": iou, "dice": dice}


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: ConusUNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = total_bce = total_dice = 0.0
    iou_list: List[float] = []
    dice_list: List[float] = []
    n = 0

    for batch in loader:
        imgs  = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(imgs)

        loss, bce_val, dice_val = seg_loss(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        b = imgs.shape[0]
        total_loss += loss.item() * b
        total_bce  += bce_val * b
        total_dice += dice_val * b
        m = compute_metrics(logits.detach(), masks)
        iou_list.append(m["iou"])
        dice_list.append(m["dice"])
        n += b

    return {
        "loss": total_loss / n,
        "bce":  total_bce  / n,
        "dice": total_dice / n,
        "iou":  float(np.mean(iou_list)),
        "dice_metric": float(np.mean(dice_list)),
    }


@torch.no_grad()
def evaluate(
    model: ConusUNet,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = total_bce = total_dice = 0.0
    iou_list: List[float] = []
    dice_list: List[float] = []
    n = 0

    for batch in loader:
        imgs  = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(imgs)

        loss, bce_val, dice_val = seg_loss(logits, masks)
        b = imgs.shape[0]
        total_loss += loss.item() * b
        total_bce  += bce_val * b
        total_dice += dice_val * b
        m = compute_metrics(logits, masks)
        iou_list.append(m["iou"])
        dice_list.append(m["dice"])
        n += b

    return {
        "loss": total_loss / n,
        "bce":  total_bce  / n,
        "dice": total_dice / n,
        "iou":  float(np.mean(iou_list)),
        "dice_metric": float(np.mean(dice_list)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    model: ConusUNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_iou: float,
    config: Dict,
    history: List[Dict],
) -> None:
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":     epoch,
        "val_iou":   val_iou,
        "config":    config,
        "history":   history,
    }, path)


def load_checkpoint(
    path: str,
    model: ConusUNet,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, float, List[Dict]]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("val_iou", 0.0), ckpt.get("history", [])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not DATASET_ROOT:
        raise ValueError("DATASET_ROOT must be set.")

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Datasets
    train_full = ConusDataset(DATASET_ROOT, target_size=(TARGET_SIZE, TARGET_SIZE), augment=True)
    val_full   = ConusDataset(DATASET_ROOT, target_size=(TARGET_SIZE, TARGET_SIZE), augment=False)

    n_val   = max(1, int(len(train_full) * VAL_SPLIT))
    n_train = len(train_full) - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_idx, val_idx = random_split(range(len(train_full)), [n_train, n_val], generator=gen)
    # Use same index split on both datasets (different augment flag)
    from torch.utils.data import Subset
    train_ds = Subset(train_full, list(train_idx))
    val_ds   = Subset(val_full,   list(val_idx))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Model
    model = ConusUNet(
        backbone=BACKBONE,
        decoder_channels=DECODER_CHANNELS,
        pretrained=bool(PRETRAINED),
    ).to(DEVICE)
    print(f"Model: {BACKBONE}  params={model.n_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    config = {
        "dataset_root":     DATASET_ROOT,
        "backbone":         BACKBONE,
        "decoder_channels": DECODER_CHANNELS,
        "pretrained":       bool(PRETRAINED),
        "target_size":      TARGET_SIZE,
        "epochs":           EPOCHS,
        "batch_size":       BATCH_SIZE,
        "lr":               LR,
        "weight_decay":     WEIGHT_DECAY,
        "val_split":        VAL_SPLIT,
        "seed":             SEED,
        "device":           str(DEVICE),
    }

    start_epoch = 1
    best_iou    = 0.0
    history: List[Dict] = []

    if MODE in ("resume", "eval") and Path(PATH_CHECKPOINT).exists():
        start_epoch, best_iou, history = load_checkpoint(
            PATH_CHECKPOINT, model, optimizer if MODE == "resume" else None
        )
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch - 1}, best IoU={best_iou:.4f}")

    if MODE == "eval":
        vl = evaluate(model, val_loader, DEVICE)
        print("Eval:", {k: f"{v:.4f}" for k, v in vl.items()})
        return

    # CSV
    csv_exists = Path(METRICS_CSV).exists() and MODE == "resume"
    csv_cols = [
        "epoch",
        "train_loss", "train_bce", "train_dice", "train_iou", "train_dice_metric",
        "val_loss",   "val_bce",   "val_dice",   "val_iou",   "val_dice_metric",
    ]
    csv_file = open(METRICS_CSV, "a" if csv_exists else "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_cols)
    if not csv_exists:
        writer.writeheader()

    for epoch in range(start_epoch, EPOCHS + 1):
        tr = train_one_epoch(model, train_loader, optimizer, DEVICE)
        vl = evaluate(model, val_loader, DEVICE)

        row = {
            "epoch":            epoch,
            "train_loss":       f"{tr['loss']:.6f}",
            "train_bce":        f"{tr['bce']:.6f}",
            "train_dice":       f"{tr['dice']:.6f}",
            "train_iou":        f"{tr['iou']:.6f}",
            "train_dice_metric":f"{tr['dice_metric']:.6f}",
            "val_loss":         f"{vl['loss']:.6f}",
            "val_bce":          f"{vl['bce']:.6f}",
            "val_dice":         f"{vl['dice']:.6f}",
            "val_iou":          f"{vl['iou']:.6f}",
            "val_dice_metric":  f"{vl['dice_metric']:.6f}",
        }
        writer.writerow(row)
        csv_file.flush()
        history.append({k: float(v) if k != "epoch" else int(v) for k, v in row.items()})

        if vl["iou"] > best_iou:
            best_iou = vl["iou"]
            save_checkpoint(PATH_CHECKPOINT, model, optimizer, epoch, best_iou, config, history)
            flag = " ✓"
        else:
            flag = ""

        print(
            f"[{epoch:3d}/{EPOCHS}]  "
            f"loss={tr['loss']:.4f}  iou={tr['iou']:.4f}  dice={tr['dice_metric']:.4f}  |  "
            f"val loss={vl['loss']:.4f}  iou={vl['iou']:.4f}  dice={vl['dice_metric']:.4f}{flag}"
        )

    csv_file.close()

    config["history"] = history
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Best val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
