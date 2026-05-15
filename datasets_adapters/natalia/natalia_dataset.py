"""
Dataset adapter for NatalIA fetal ultrasound cine frames.

For foundation-model (MAE) pre-training: walks every exam subfolder under ``root``
and exposes a flat list of all cine-frame JPEGs.

Images are returned at their **native** resolution as ``[1, H, W]`` grayscale
tensors. The MAE pipeline applies the max-height shrink + per-batch padding
(:func:`embeddings.vit.train.apply_max_height_shrink`,
:func:`embeddings.vit.train.mae_pad_collate`), so this dataset performs no
sizing transforms of its own.

Dataset structure (see readme.md):
    root/
        resume.csv
        metadata.csv
        <EXAM_NAME_1>/
            <CINEMAFRAME_1>.jpeg
            <CINEMAFRAME_2>.jpeg
            ...
        <EXAM_NAME_2>/
            ...

Source: https://zenodo.org/records/14193949
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


_IMAGE_EXTS = (".jpeg", ".jpg", ".png")


class NataliaDataset(Dataset):
    """
    NatalIA cine-frame dataset for unsupervised pre-training.

    Args:
        root: Root directory containing one subfolder per exam.
        transform: Optional callable taking a PIL image and returning a tensor.
                   If ``None``, returns ``[1, H, W]`` grayscale float tensor.
        exam_names: Optional list of exam folder names to restrict to.
                    Useful for train/val splits driven by an external CSV.
    """

    def __init__(
        self,
        root: str,
        *,
        transform: Optional[Callable] = None,
        exam_names: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root}")

        if exam_names is not None:
            exam_dirs = [self.root / name for name in exam_names]
        else:
            exam_dirs = sorted(
                p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")
            )

        self.image_paths: List[Path] = []
        self.image_ids: List[str] = []

        for exam_dir in exam_dirs:
            if not exam_dir.is_dir():
                continue
            for p in sorted(exam_dir.iterdir()):
                if p.suffix.lower() in _IMAGE_EXTS:
                    self.image_paths.append(p)
                    self.image_ids.append(f"{exam_dir.name}/{p.stem}")

        if len(self.image_paths) == 0:
            raise ValueError(f"No image files found under {self.root}")

        print(
            f"Loaded {len(self.image_paths)} frames from "
            f"{len(set(p.parent for p in self.image_paths))} exams in {self.root}"
        )

        self.transform = transform if transform is not None else self._default_transform

    @staticmethod
    def _default_transform(img: Image.Image) -> torch.Tensor:
        """PIL grayscale → ``[1, H, W]`` float tensor in [0, 1]. No resizing."""
        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)
        if img.dim() == 2:
            img = img.unsqueeze(0)
        if img.size(0) > 1:
            img = img.mean(dim=0, keepdim=True)
        return img

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("L")
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}") from e
        return {
            "image": self.transform(img),
            "image_id": self.image_ids[idx],
        }
