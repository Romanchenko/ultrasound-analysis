"""
Segmentation baseline: frozen encoder + lightweight FCN decoder on abdominal dataset.

Encoder types: "mae", "vicreg", "timm" (set via ENCODER_TYPE).
Trains decoder only; evaluates per-class and mean IoU/Dice each epoch.

Environment variables (all optional, with defaults):
    DIR_ABDOMINAL            root dir of abdominal segmentation dataset
    ENCODER_TYPE             "mae" | "vicreg" | "timm"  (default: "mae")
    PATH_ENCODER_CHECKPOINT  path to MAE or VICReg .pt checkpoint
    TIMM_MODEL               timm model name (used when ENCODER_TYPE="timm")
    EPOCHS                   50
    BATCH_SIZE               16
    LR                       1e-3
    WEIGHT_DECAY             1e-4
    VAL_SPLIT                0.2
    NUM_WORKERS              4
    TARGET_SIZE              224  (square H=W)
    DECODER_HIDDEN_DIM       256
    DECODER_DEPTH            2
    SEED                     42
    MODE                     train | resume | eval
    DEVICE                   auto
    RANDOM_ENCODER           0    (1 = reinitialise encoder weights)
    PATH_HEAD_CHECKPOINT     best_seg_head.pt
    METRICS_CSV              seg_metrics.csv
    CONFIG_PATH              seg_run_config.json
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets_adapters.abdominal_segmentation.abdominal_dataset import (
    AbdominalSegmentationDataset,
    STRUCTURE_KEYS,
)

NUM_CLASSES = len(STRUCTURE_KEYS)  # 4

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DIR_ABDOMINAL           = os.environ.get("DIR_ABDOMINAL", "")
ENCODER_TYPE            = os.environ.get("ENCODER_TYPE", "mae").strip().lower()
PATH_ENCODER_CHECKPOINT = os.environ.get("PATH_ENCODER_CHECKPOINT", "").strip()
TIMM_MODEL              = os.environ.get("TIMM_MODEL", "").strip()

EPOCHS           = int(os.environ.get("EPOCHS", 50))
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE", 16))
LR               = float(os.environ.get("LR", 1e-3))
WEIGHT_DECAY     = float(os.environ.get("WEIGHT_DECAY", 1e-4))
VAL_SPLIT        = float(os.environ.get("VAL_SPLIT", 0.2))
NUM_WORKERS      = int(os.environ.get("NUM_WORKERS", 4))
TARGET_SIZE      = int(os.environ.get("TARGET_SIZE", 224))
DECODER_HIDDEN_DIM = int(os.environ.get("DECODER_HIDDEN_DIM", 256))
DECODER_DEPTH    = int(os.environ.get("DECODER_DEPTH", 2))
SEED             = int(os.environ.get("SEED", 42))
MODE             = os.environ.get("MODE", "train").strip().lower()
RANDOM_ENCODER   = int(os.environ.get("RANDOM_ENCODER", 0))

PATH_HEAD_CHECKPOINT = os.environ.get("PATH_HEAD_CHECKPOINT", "best_seg_head.pt")
METRICS_CSV          = os.environ.get("METRICS_CSV", "seg_metrics.csv")
CONFIG_PATH          = os.environ.get("CONFIG_PATH", "seg_run_config.json")

_device_env = os.environ.get("DEVICE", "").strip()
DEVICE = torch.device(_device_env if _device_env else ("cuda" if torch.cuda.is_available() else "cpu"))


# ─────────────────────────────────────────────────────────────────────────────
# Model components
# ─────────────────────────────────────────────────────────────────────────────

class FCNDecoder(nn.Module):
    """
    Lightweight FCN segmentation head.

    Input:  [B, embed_dim, grid_h, grid_w]
    Output: [B, num_classes, H, W]  where H = grid_h * patch_size
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        patch_size: int,
        depth: int = 2,
    ):
        super().__init__()
        self.patch_size = patch_size

        layers: List[nn.Module] = [
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 1):
            layers += [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.Upsample(scale_factor=patch_size, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SegmentationModel(nn.Module):
    """Frozen encoder + trainable FCN decoder."""

    def __init__(self, encoder: nn.Module, decoder: FCNDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        imgs: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            feat = self.encoder.encode_patches(imgs, pad_mask)  # [B, D, h, w]
        return self.decoder(feat)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_encoder(
    encoder_type: str,
    checkpoint: str,
    timm_model: str,
    device: torch.device,
) -> nn.Module:
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

    if RANDOM_ENCODER:
        print("Re-initialising encoder weights (RANDOM_ENCODER=1)")
        if hasattr(enc, '_init_weights'):
            enc._init_weights()
        else:
            for m in enc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


# ─────────────────────────────────────────────────────────────────────────────
# Loss and metrics
# ─────────────────────────────────────────────────────────────────────────────

def seg_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, float, float]:
    """BCE + Dice loss. Returns (total, bce_val, dice_val)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)

    prob = torch.sigmoid(logits)
    eps = 1e-6
    inter = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1.0 - ((2.0 * inter + eps) / (union + eps)).mean()

    total = bce + dice
    return total, bce.item(), dice.item()


def compute_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Per-class and mean IoU / Dice. Returns flat dict."""
    pred = (torch.sigmoid(logits) > threshold).float()
    metrics: Dict[str, float] = {}
    iou_list, dice_list = [], []

    for c in range(logits.shape[1]):
        p = pred[:, c]
        t = target[:, c]
        inter = (p * t).sum().item()
        union = (p + t).clamp_max(1).sum().item()
        iou = inter / (union + 1e-6)
        dice = (2 * inter) / (p.sum().item() + t.sum().item() + 1e-6)
        metrics[f"iou_class{c}"] = iou
        metrics[f"dice_class{c}"] = dice
        iou_list.append(iou)
        dice_list.append(dice)

    metrics["mean_iou"]  = float(np.mean(iou_list))
    metrics["mean_dice"] = float(np.mean(dice_list))
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Mask padding collate
# ─────────────────────────────────────────────────────────────────────────────

def _pad_mask(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Pad mask channel dim to num_classes with zeros if needed."""
    c = mask.shape[0]
    if c < num_classes:
        pad = torch.zeros(num_classes - c, *mask.shape[1:], dtype=mask.dtype)
        mask = torch.cat([mask, pad], dim=0)
    return mask[:num_classes]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    images = torch.stack([s["image"] for s in batch])
    masks  = torch.stack([_pad_mask(s["mask"], NUM_CLASSES) for s in batch])
    return {"image": images, "mask": masks}


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: SegmentationModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.decoder.train()
    total_loss = total_bce = total_dice = 0.0
    all_iou: List[float] = []
    all_dice: List[float] = []
    n = 0

    for batch in loader:
        imgs  = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(imgs)

        if logits.shape[2:] != masks.shape[2:]:
            logits = F.interpolate(logits, size=masks.shape[2:], mode="bilinear", align_corners=False)

        loss, bce_val, dice_val = seg_loss(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        b = imgs.shape[0]
        total_loss += loss.item() * b
        total_bce  += bce_val * b
        total_dice += dice_val * b
        m = compute_metrics(logits.detach(), masks)
        all_iou.append(m["mean_iou"])
        all_dice.append(m["mean_dice"])
        n += b

    return {
        "loss":      total_loss / n,
        "bce":       total_bce  / n,
        "dice":      total_dice / n,
        "mean_iou":  float(np.mean(all_iou)),
        "mean_dice": float(np.mean(all_dice)),
    }


@torch.no_grad()
def evaluate(
    model: SegmentationModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.decoder.eval()
    total_loss = total_bce = total_dice = 0.0
    all_iou: List[float] = []
    all_dice: List[float] = []
    n = 0

    for batch in loader:
        imgs  = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(imgs)

        if logits.shape[2:] != masks.shape[2:]:
            logits = F.interpolate(logits, size=masks.shape[2:], mode="bilinear", align_corners=False)

        loss, bce_val, dice_val = seg_loss(logits, masks)
        b = imgs.shape[0]
        total_loss += loss.item() * b
        total_bce  += bce_val * b
        total_dice += dice_val * b
        m = compute_metrics(logits, masks)
        all_iou.append(m["mean_iou"])
        all_dice.append(m["mean_dice"])
        n += b

    return {
        "loss":      total_loss / n,
        "bce":       total_bce  / n,
        "dice":      total_dice / n,
        "mean_iou":  float(np.mean(all_iou)),
        "mean_dice": float(np.mean(all_dice)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(
    path: str,
    decoder: FCNDecoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_mean_iou: float,
    config: Dict,
    history: List[Dict],
) -> None:
    torch.save({
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_mean_iou": val_mean_iou,
        "config": config,
        "history": history,
    }, path)


def _load_head_checkpoint(
    path: str,
    decoder: FCNDecoder,
    optimizer: Optional[torch.optim.Optimizer],
) -> Tuple[int, float, List[Dict]]:
    ckpt = torch.load(path, map_location="cpu")
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("val_mean_iou", 0.0), ckpt.get("history", [])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not DIR_ABDOMINAL:
        raise ValueError("DIR_ABDOMINAL must be set.")
    if ENCODER_TYPE not in ("mae", "vicreg", "timm"):
        raise ValueError(f"ENCODER_TYPE must be mae|vicreg|timm, got {ENCODER_TYPE!r}")
    if ENCODER_TYPE in ("mae", "vicreg") and not PATH_ENCODER_CHECKPOINT:
        raise ValueError(f"PATH_ENCODER_CHECKPOINT required for ENCODER_TYPE={ENCODER_TYPE!r}")
    if ENCODER_TYPE == "timm" and not TIMM_MODEL:
        raise ValueError("TIMM_MODEL must be set when ENCODER_TYPE='timm'")

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Load encoder
    encoder = _load_encoder(ENCODER_TYPE, PATH_ENCODER_CHECKPOINT, TIMM_MODEL, DEVICE)
    embed_dim = encoder.embed_dim
    patch_size = encoder.patch_size

    # Dataset
    dataset = AbdominalSegmentationDataset(
        root=DIR_ABDOMINAL,
        load_masks=True,
        target_size=(TARGET_SIZE, TARGET_SIZE),
    )

    n_val  = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
    )

    # Model
    decoder = FCNDecoder(embed_dim, DECODER_HIDDEN_DIM, NUM_CLASSES, patch_size, depth=DECODER_DEPTH)
    decoder.to(DEVICE)
    model = SegmentationModel(encoder, decoder).to(DEVICE)

    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    config = {
        "dir_abdominal":           DIR_ABDOMINAL,
        "encoder_type":            ENCODER_TYPE,
        "path_encoder_checkpoint": PATH_ENCODER_CHECKPOINT,
        "timm_model":              TIMM_MODEL,
        "embed_dim":               embed_dim,
        "patch_size":              patch_size,
        "decoder_hidden_dim":      DECODER_HIDDEN_DIM,
        "decoder_depth":           DECODER_DEPTH,
        "num_classes":             NUM_CLASSES,
        "structure_keys":          STRUCTURE_KEYS,
        "epochs":                  EPOCHS,
        "batch_size":              BATCH_SIZE,
        "lr":                      LR,
        "weight_decay":            WEIGHT_DECAY,
        "val_split":               VAL_SPLIT,
        "target_size":             TARGET_SIZE,
        "seed":                    SEED,
        "random_encoder":          bool(RANDOM_ENCODER),
        "device":                  str(DEVICE),
    }

    start_epoch = 1
    best_iou = 0.0
    history: List[Dict] = []

    if MODE in ("resume", "eval") and Path(PATH_HEAD_CHECKPOINT).exists():
        start_epoch, best_iou, history = _load_head_checkpoint(
            PATH_HEAD_CHECKPOINT, decoder, optimizer if MODE == "resume" else None
        )
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch - 1}, best IoU={best_iou:.4f}")

    if MODE == "eval":
        val_metrics = evaluate(model, val_loader, DEVICE)
        print("Eval:", {k: f"{v:.4f}" for k, v in val_metrics.items()})
        return

    # CSV header
    csv_exists = Path(METRICS_CSV).exists() and MODE == "resume"
    csv_file = open(METRICS_CSV, "a" if csv_exists else "w", newline="")
    csv_cols = [
        "epoch",
        "train_loss", "train_bce", "train_dice", "train_mean_iou", "train_mean_dice",
        "val_loss",   "val_bce",   "val_dice",   "val_mean_iou",   "val_mean_dice",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=csv_cols)
    if not csv_exists:
        writer.writeheader()

    for epoch in range(start_epoch, EPOCHS + 1):
        tr = train_one_epoch(model, train_loader, optimizer, DEVICE)
        vl = evaluate(model, val_loader, DEVICE)

        row = {
            "epoch":          epoch,
            "train_loss":     f"{tr['loss']:.6f}",
            "train_bce":      f"{tr['bce']:.6f}",
            "train_dice":     f"{tr['dice']:.6f}",
            "train_mean_iou": f"{tr['mean_iou']:.6f}",
            "train_mean_dice":f"{tr['mean_dice']:.6f}",
            "val_loss":       f"{vl['loss']:.6f}",
            "val_bce":        f"{vl['bce']:.6f}",
            "val_dice":       f"{vl['dice']:.6f}",
            "val_mean_iou":   f"{vl['mean_iou']:.6f}",
            "val_mean_dice":  f"{vl['mean_dice']:.6f}",
        }
        writer.writerow(row)
        csv_file.flush()

        history.append({k: float(v) if k != "epoch" else int(v) for k, v in row.items()})

        if vl["mean_iou"] > best_iou:
            best_iou = vl["mean_iou"]
            _save_checkpoint(PATH_HEAD_CHECKPOINT, decoder, optimizer, epoch, best_iou, config, history)
            flag = " ✓"
        else:
            flag = ""

        print(
            f"[{epoch:3d}/{EPOCHS}]  "
            f"train loss={tr['loss']:.4f} iou={tr['mean_iou']:.4f}  |  "
            f"val loss={vl['loss']:.4f} iou={vl['mean_iou']:.4f}{flag}"
        )

    csv_file.close()

    # Final per-class metrics on val set
    final_metrics = evaluate(model, val_loader, DEVICE)
    with torch.no_grad():
        all_logits, all_masks = [], []
        for batch in val_loader:
            imgs  = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            logits = model(imgs)
            if logits.shape[2:] != masks.shape[2:]:
                logits = F.interpolate(logits, size=masks.shape[2:], mode="bilinear", align_corners=False)
            all_logits.append(logits.cpu())
            all_masks.append(masks.cpu())
    per_class = compute_metrics(torch.cat(all_logits), torch.cat(all_masks))

    config["history"] = history
    config["final_val_metrics"] = per_class
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Best val mean IoU: {best_iou:.4f}")
    print(f"Per-class IoU: " + ", ".join(
        f"{STRUCTURE_KEYS[c]}={per_class[f'iou_class{c}']:.4f}" for c in range(NUM_CLASSES)
    ))


if __name__ == "__main__":
    main()
