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

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


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

    Images are returned at their **native** resolution. The MAE pipeline takes
    care of the max-height shrink + per-batch padding
    (:func:`embeddings.vit.train.apply_max_height_shrink`,
    :func:`embeddings.vit.train.mae_pad_collate`), so this dataset performs no
    sizing transforms of its own.

    Use for:
    - **Embeddings (MAE)**: load_masks=False, returns {'image': [1,H,W]}
    - **Segmentation**: load_masks=True, returns {'image': [1,H,W], 'mask': [C,H,W]}

    Args:
        root: Root directory containing training/, validation/, testing/
        split: 'training', 'validation', 'testing' (or 'train', 'val', 'test')
        load_masks: If True, load segmentation masks from annfiles_mask/
        mask_target: 'cardiac', 'thorax', or 'both'.
                     'both' returns mask [2,H,W] with cardiac in ch0, thorax in ch1.
        transform: Optional callable. If ``None``, the image is returned as a
                   grayscale ``[1, H, W]`` tensor (via ``ToTensor``) with no
                   resizing. When ``load_masks=True``, the transform receives a
                   dict with ``'image'`` and ``'mask'`` and must return a dict
                   with the same keys.
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
        transform: Optional[Callable] = None,
        images_dir: str = "images",
        masks_dir: str = "annfiles_mask",
        target_size: Optional[Any] = None,  # deprecated, ignored
        preprocessed: bool = False,
    ):
        if target_size is not None:
            warnings.warn(
                "FOCUSDataset(target_size=...) is deprecated and ignored: "
                "images are now returned at their native resolution and sized by "
                "the MAE pipeline (apply_max_height_shrink + mae_pad_collate).",
                DeprecationWarning, stacklevel=2,
            )
        self.root = Path(root)
        self.split_dir = _SPLIT_DIR_MAP[split]
        self.load_masks = load_masks
        self.mask_target = mask_target
        self.images_dir_name = images_dir
        self.masks_dir_name = masks_dir
        self.preprocessed = preprocessed

        split_path = self.root / self.split_dir
        if preprocessed:
            self.images_dir = split_path / (images_dir + '_preprocessed')
        else:
            self.images_dir = split_path / images_dir
        self.masks_dir = split_path / masks_dir

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

    def _get_default_transform(self) -> Callable:
        """Default transform: PIL grayscale → ``[1, H, W]`` tensor, no resizing.

        When ``load_masks=True`` the dataset builds the masks itself (already
        ``[C, H, W]`` tensors at native resolution) and the default transform
        just normalises the image to a single-channel float tensor.
        """
        def _to_grayscale_tensor(img) -> torch.Tensor:
            if not isinstance(img, torch.Tensor):
                img = T.ToTensor()(img)
            if img.dim() == 2:
                img = img.unsqueeze(0)
            if img.size(0) > 1:
                img = img.mean(dim=0, keepdim=True)
            return img

        if self.load_masks:
            def transform_fn(sample: Dict) -> Dict:
                return {
                    "image": _to_grayscale_tensor(sample["image"]),
                    "mask": sample["mask"],
                }
            return transform_fn
        return _to_grayscale_tensor

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
