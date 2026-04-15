#!/usr/bin/env python
"""
Train a CycleGAN on a pair of datasets (A, B), then translate every image
from domain B into domain A and save the results to disk.

Usage (from the project root):

    python -m domain_adaptation.cyclegan.train_and_translate \\
        --config domain_adaptation/cyclegan/pair_config.json

Optional CLI overrides:

    --checkpoint_dir ./my_ckpts
    --output_dir     ./my_translated
    --skip_training              # only run the translation phase
"""

import sys
import json
import argparse
import importlib
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root on sys.path so that adapter imports work.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain_adaptation.cyclegan.model import create_cyclegan_model   # noqa: E402
from domain_adaptation.cyclegan.train import CycleGANTrainer         # noqa: E402
from domain_adaptation.cyclegan.dataloaders.base_dataloader import ( # noqa: E402
    UnpairedDataset,
)


# ===================================================================
# Helpers
# ===================================================================

def resolve_class(dotpath: str):
    """Import and return the class at *dotpath* (e.g. ``pkg.mod.Class``)."""
    module_path, class_name = dotpath.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def make_base_transform(image_size: int):
    """Resize + ToTensor ([0, 1]) — injected into every adapter so that its
    own default transforms (which may require optional packages) are bypassed."""
    return [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]


# ===================================================================
# Dataset wrappers
# ===================================================================

class CycleGANImageWrapper(Dataset):
    """Wraps any adapter dataset for CycleGAN **training**.

    * Extracts the ``'image'`` tensor from the adapter's dict output.
    * Adjusts the number of channels to *target_channels*.
    * Normalises [0, 1] → [-1, 1].
    """

    def __init__(self, adapter: Dataset, target_channels: int = 1):
        self.adapter = adapter
        self.target_channels = target_channels

    def __len__(self):
        return len(self.adapter)

    def __getitem__(self, idx):
        sample = self.adapter[idx]
        img = sample["image"] if isinstance(sample, dict) else sample

        if img.dim() == 2:
            img = img.unsqueeze(0)

        c = img.size(0)
        if self.target_channels == 1 and c > 1:
            img = img.mean(dim=0, keepdim=True)
        elif self.target_channels == 3 and c == 1:
            img = img.repeat(3, 1, 1)

        # [0, 1] → [-1, 1]
        img = img * 2.0 - 1.0
        return img


class TranslationDataset(Dataset):
    """Wraps an adapter for the **translation** phase.

    Returns ``(image_tensor, filename_stem)`` so that the caller can
    save each translated image under a meaningful name.
    """

    def __init__(self, adapter: Dataset, target_channels: int = 1):
        self.adapter = adapter
        self.target_channels = target_channels

    def __len__(self):
        return len(self.adapter)

    def __getitem__(self, idx):
        sample = self.adapter[idx]
        img = sample["image"] if isinstance(sample, dict) else sample

        if img.dim() == 2:
            img = img.unsqueeze(0)

        c = img.size(0)
        if self.target_channels == 1 and c > 1:
            img = img.mean(dim=0, keepdim=True)
        elif self.target_channels == 3 and c == 1:
            img = img.repeat(3, 1, 1)

        # [0, 1] → [-1, 1]
        img = img * 2.0 - 1.0

        # Best-effort filename from the adapter
        fname = self._extract_filename(idx, sample)
        return img, fname

    # ------------------------------------------------------------------
    def _extract_filename(self, idx, sample):
        """Try several conventions used by the existing adapters."""
        # Adapters keep image_paths as a list of Path objects
        if hasattr(self.adapter, "image_paths"):
            return self.adapter.image_paths[idx].stem

        # Fallback: dig into sample metadata
        if isinstance(sample, dict):
            for key in ("metadata", "label"):
                meta = sample.get(key)
                if isinstance(meta, dict):
                    for name_key in ("Image_name", "filename", "image_name"):
                        if name_key in meta:
                            return str(meta[name_key]).replace(".png", "")

        return f"image_{idx:06d}"


# ===================================================================
# Adapter factory
# ===================================================================

def build_adapter(cfg: dict, image_size: int) -> Dataset:
    """Instantiate a dataset adapter from its config block."""
    cls = resolve_class(cfg["adapter"])
    params = dict(cfg.get("params", {}))

    # Inject a standard transform to bypass adapter defaults.
    params.setdefault("transform", make_base_transform(image_size))
    params.setdefault("target_size", (image_size, image_size))

    return cls(**params)


# ===================================================================
# Phase 1 — Training
# ===================================================================

def train(
    ds_a: Dataset,
    ds_b: Dataset,
    training_cfg: dict,
    checkpoint_dir: Path,
    device: torch.device,
):
    """Train CycleGAN and save **only** the best checkpoint."""

    channels        = training_cfg.get("input_channels", 1)
    n_epochs        = training_cfg.get("n_epochs", 200)
    batch_size      = training_cfg.get("batch_size", 4)
    num_workers     = training_cfg.get("num_workers", 4)
    lr_g            = training_cfg.get("lr_g", 2e-4)
    lr_d            = training_cfg.get("lr_d", 2e-4)
    lambda_cycle    = training_cfg.get("lambda_cycle", 10.0)
    lambda_identity = training_cfg.get("lambda_identity", 0.5)
    n_residual      = training_cfg.get("n_residual_blocks", 9)
    vis_every       = training_cfg.get("visualize_every", 10)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = checkpoint_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Wrap adapters
    wrapped_a = CycleGANImageWrapper(ds_a, target_channels=channels)
    wrapped_b = CycleGANImageWrapper(ds_b, target_channels=channels)

    unpaired = UnpairedDataset(wrapped_a, wrapped_b)
    loader = DataLoader(
        unpaired,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Model + trainer
    model = create_cyclegan_model(
        input_channels_a=channels,
        input_channels_b=channels,
        n_residual_blocks=n_residual,
    )

    trainer = CycleGANTrainer(
        model=model,
        device=device,
        lambda_cycle=lambda_cycle,
        lambda_identity=lambda_identity,
        lr_g=lr_g,
        lr_d=lr_d,
    )

    # Training loop — best-only saving
    best_loss = float("inf")

    for epoch in range(n_epochs):
        losses = trainer.train_epoch(loader, epoch, n_epochs)

        g_loss = losses["loss_G"]
        print(
            f"Epoch {epoch + 1}/{n_epochs}  "
            f"G={g_loss:.4f}  "
            f"D_A={losses['loss_D_A']:.4f}  "
            f"D_B={losses['loss_D_B']:.4f}  "
            f"Cycle={losses['loss_cycle']:.4f}  "
            f"Idt={losses['loss_identity']:.4f}"
        )

        if (epoch + 1) % vis_every == 0 or epoch == 0:
            trainer.visualize_samples(
                loader, epoch, str(samples_dir), num_samples=1
            )

        if g_loss < best_loss:
            best_loss = g_loss
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "training_cfg": training_cfg,
                    "model_state_dict": model.state_dict(),
                    "optimizer_G": trainer.optimizer_G.state_dict(),
                    "optimizer_D_A": trainer.optimizer_D_A.state_dict(),
                    "optimizer_D_B": trainer.optimizer_D_B.state_dict(),
                    "history": trainer.history,
                },
                best_path,
            )
            print(f"  -> Best model saved (G loss {best_loss:.4f}) -> {best_path}")

    print(f"\nTraining complete.  Best G loss: {best_loss:.4f}")


# ===================================================================
# Phase 2 — Translate all B images → A domain
# ===================================================================

def translate(
    ds_b: Dataset,
    training_cfg: dict,
    checkpoint_dir: Path,
    output_dir: Path,
    device: torch.device,
):
    """Load the best checkpoint and translate every image in *ds_b*."""

    channels   = training_cfg.get("input_channels", 1)
    n_residual = training_cfg.get("n_residual_blocks", 9)
    batch_size = training_cfg.get("batch_size", 4)
    num_workers = training_cfg.get("num_workers", 4)

    best_path = checkpoint_dir / "best_model.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found at {best_path}")

    # Rebuild model and load weights
    model = create_cyclegan_model(
        input_channels_a=channels,
        input_channels_b=channels,
        n_residual_blocks=n_residual,
    )
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(
        f"Loaded best checkpoint from epoch {ckpt['epoch'] + 1} "
        f"(G loss {ckpt['best_loss']:.4f})"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    trans_ds = TranslationDataset(ds_b, target_channels=channels)
    loader = DataLoader(
        trans_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    saved = 0
    with torch.no_grad():
        for imgs, fnames in tqdm(loader, desc="Translating B → A"):
            imgs = imgs.to(device)
            translated = model.G_B2A(imgs)

            # [-1, 1] → [0, 1]
            translated = (translated + 1.0) / 2.0
            translated = translated.clamp(0, 1).cpu()

            for tensor, fname in zip(translated, fnames):
                out_path = output_dir / f"{fname}.png"
                save_image(tensor, str(out_path))
                saved += 1

    print(f"Saved {saved} translated images to {output_dir}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train CycleGAN on a dataset pair, then translate B → A.",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to JSON config (see pair_config.json for format).",
    )
    parser.add_argument(
        "--checkpoint_dir", default=None,
        help="Override checkpoint directory from config.",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Override translated-image output directory from config.",
    )
    parser.add_argument(
        "--skip_training", action="store_true",
        help="Skip training and only run the translation phase "
             "(expects best_model.pt in checkpoint_dir).",
    )
    args = parser.parse_args()

    # ---- load config ----
    with open(args.config) as f:
        config = json.load(f)

    training_cfg = config.get("training", {})
    image_size   = training_cfg.get("image_size", 256)

    checkpoint_dir = Path(args.checkpoint_dir or config["checkpoint_dir"])
    output_dir     = Path(args.output_dir or config["output_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ---- build adapters ----
    print("Building dataset A …")
    ds_a = build_adapter(config["dataset_a"], image_size)
    print(f"Building dataset B …")
    ds_b = build_adapter(config["dataset_b"], image_size)

    # ---- Phase 1: train ----
    if not args.skip_training:
        print(f"\n{'=' * 50}")
        print("Phase 1: Training CycleGAN")
        print(f"{'=' * 50}\n")
        train(ds_a, ds_b, training_cfg, checkpoint_dir, device)
    else:
        print("Skipping training (--skip_training).\n")

    # ---- Phase 2: translate B → A ----
    print(f"\n{'=' * 50}")
    print("Phase 2: Translating dataset B → domain A")
    print(f"{'=' * 50}\n")
    translate(ds_b, training_cfg, checkpoint_dir, output_dir, device)

    print("\nDone.")


if __name__ == "__main__":
    main()



