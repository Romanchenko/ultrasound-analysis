"""
Dataset adapter for FOCUS: Four-chamber Ultrasound Image Dataset for Fetal Cardiac Biometric Measurement.

Supports:
- Embeddings training (MAE): returns grayscale images [1, H, W]
- Segmentation validation: returns image + masks (cardiac, thorax, or both)

Dataset structure (see readme.md):
    root/
        training/   (or training/)
            images/          001.png, 002.png, ...
            annfiles_mask/   001-cardiac.png, 001-thorax.png, ...
            annfiles_ellipse/
            annfiles_rectangle/
        validation/
            ...
        testing/
            ...

Links:
- https://github.com/szuboy/FOCUS-dataset
- https://zenodo.org/records/14597550
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from embeddings.vit.train import resize_keep_aspect_pad


SplitName = Literal["training", "validation", "testing", "train", "val", "test"]
MaskTarget = Literal["cardiac", "thorax", "both"]


_SPLIT_DIR_MAP = {
    "train": "training",
    "val": "validation",
    "test": "testing",
    "training": "training",
    "validation": "validation",
    "testing": "testing",
}


class FOCUSDataset(Dataset):
    """
    FOCUS dataset for fetal four-chamber ultrasound images.

    Use for:
    - **Embeddings (MAE)**: load_masks=False, returns {'image': [1,H,W]}
    - **Segmentation**: load_masks=True, returns {'image': [1,H,W], 'mask': [C,H,W]}

    Args:
        root: Root directory containing training/, validation/, testing/
        split: 'training', 'validation', 'testing' (or 'train', 'val', 'test')
        load_masks: If True, load segmentation masks from annfiles_mask/
        mask_target: 'cardiac', 'thorax', or 'both'.
                     'both' returns mask [2,H,W] with cardiac in ch0, thorax in ch1.
        transform: Optional transform. If None, uses default resize to target_size.
                   When load_masks=True, transform receives dict with 'image' and 'mask'
                   and must return dict with same keys.
        target_size: (H, W) for resize. Default (224, 224).
        images_dir: Name of images subdir within each split (default 'images')
        masks_dir: Name of masks subdir (default 'annfiles_mask')
    """

    def __init__(
        self,
        root: str,
        split: SplitName = "training",
        *,
        load_masks: bool = False,
        mask_target: MaskTarget = "both",
        transform: Optional[T.Compose] = None,
        target_size: Tuple[int, int] = (224, 224),
        images_dir: str = "images",
        masks_dir: str = "annfiles_mask",
    ):
        self.root = Path(root)
        self.split_dir = _SPLIT_DIR_MAP[split]
        self.load_masks = load_masks
        self.mask_target = mask_target
        self.target_size = target_size
        self.images_dir_name = images_dir
        self.masks_dir_name = masks_dir

        self.images_dir = self.root / self.split_dir / images_dir
        self.masks_dir = self.root / self.split_dir / masks_dir

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        # Collect image IDs (filename without extension)
        self.image_paths: List[Path] = []
        self.image_ids: List[str] = []

        for p in sorted(self.images_dir.iterdir()):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                self.image_paths.append(p)
                self.image_ids.append(p.stem)

        if load_masks and not self.masks_dir.exists():
            raise FileNotFoundError(
                f"Masks directory not found: {self.masks_dir}. "
                "Set load_masks=False for embeddings-only mode."
            )

        # Filter to images that have masks when load_masks=True
        if load_masks:
            valid_paths = []
            valid_ids = []
            for path, img_id in zip(self.image_paths, self.image_ids):
                cardiac_path = self.masks_dir / f"{img_id}-cardiac.png"
                thorax_path = self.masks_dir / f"{img_id}-thorax.png"
                needed = []
                if mask_target in ("cardiac", "both"):
                    needed.append(cardiac_path)
                if mask_target in ("thorax", "both"):
                    needed.append(thorax_path)
                if all(p.exists() for p in needed):
                    valid_paths.append(path)
                    valid_ids.append(img_id)
                else:
                    pass  # skip images without required masks
            self.image_paths = valid_paths
            self.image_ids = valid_ids

        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in {self.images_dir}")

        print(f"Loaded {len(self.image_paths)} images from {self.root / self.split_dir}")

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()

    def _get_default_transform(self):
        """Default: resize image (and mask) to target_size, aspect-ratio preserved."""
        size = self.target_size[0]

        if self.load_masks:

            def transform_fn(sample: Dict) -> Dict:
                img = sample["image"]
                mask = sample["mask"]
                if not isinstance(img, torch.Tensor):
                    img = T.ToTensor()(img)
                if img.dim() == 2:
                    img = img.unsqueeze(0)
                if img.size(0) > 1:
                    img = img.mean(dim=0, keepdim=True)
                img = resize_keep_aspect_pad(
                    img, size, interpolation=T.InterpolationMode.BILINEAR
                )
                # Resize mask with nearest to preserve labels
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                new_h, new_w = img.shape[1], img.shape[2]
                mask = F.interpolate(
                    mask.unsqueeze(0).float(),
                    size=(new_h, new_w),
                    mode="nearest",
                ).squeeze(0)
                return {"image": img, "mask": mask}

            return transform_fn
        else:

            def resize_img(img) -> torch.Tensor:
                if not isinstance(img, torch.Tensor):
                    img = T.ToTensor()(img)
                if img.dim() == 2:
                    img = img.unsqueeze(0)
                if img.size(0) > 1:
                    img = img.mean(dim=0, keepdim=True)
                return resize_keep_aspect_pad(
                    img, size, interpolation=T.InterpolationMode.BILINEAR
                )

            return resize_img

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        image_path = self.image_paths[idx]
        image_id = self.image_ids[idx]

        # Load image as grayscale
        try:
            image = Image.open(image_path).convert("L")
        except Exception as e:
            raise RuntimeError(f"Failed to load {image_path}: {e}") from e

        if not self.load_masks:
            # Embeddings mode: return image only
            if callable(self.transform):
                image = self.transform(image)
            else:
                image = T.ToTensor()(image)
                if image.dim() == 2:
                    image = image.unsqueeze(0)
            return {"image": image}

        # Segmentation mode: load masks
        masks = []
        if self.mask_target in ("cardiac", "both"):
            cardiac_path = self.masks_dir / f"{image_id}-cardiac.png"
            m = Image.open(cardiac_path).convert("L")
            masks.append(T.ToTensor()(m))
        if self.mask_target in ("thorax", "both"):
            thorax_path = self.masks_dir / f"{image_id}-thorax.png"
            m = Image.open(thorax_path).convert("L")
            masks.append(T.ToTensor()(m))

        # Binarize masks (ToTensor gives [0,1]; threshold at 0.5)
        masks = [(m > 0.5).float() for m in masks]
        mask = torch.cat(masks, dim=0)  # [1,H,W] or [2,H,W]

        sample = {"image": image, "mask": mask}
        sample = self.transform(sample)
        sample["image_id"] = image_id
        return sample
