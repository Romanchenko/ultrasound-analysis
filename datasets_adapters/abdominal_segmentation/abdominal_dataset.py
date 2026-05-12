"""
Dataset adapter for Fetal Abdominal Structures Segmentation (Mendeley 4gcpm9dsc3).

Loads images from .npy files. Each npy may contain:
- A raw numpy array (image data), or
- A dict mapping structure names to mask arrays (and optionally 'image' key).

Supports:
- Embeddings training (MAE): returns grayscale images [1, H, W]
- Segmentation validation: returns image + combined mask [C, H, W]

Dataset structure (see readme.md):
    root/
        ARRAY_FORMAT/
            P01_IMG1.npy
            P01_IMG2.npy
        IMAGES/
            P01_IMG1.png
            P01_IMG2.png

Source: https://data.mendeley.com/datasets/4gcpm9dsc3/1
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from embeddings.vit.train import resize_keep_aspect_pad


# Default structure keys in npy dict (dataset: abdominal aorta, umbilical vein, stomach, liver)
# Adjust structure_keys if your npy uses different key names
STRUCTURE_KEYS = [
    "abdominal_aorta",
    "intrahepatic_umbilical_vein",
    "stomach",
    "liver",
]


class AbdominalSegmentationDataset(Dataset):
    """
    Fetal abdominal structures segmentation dataset loading from .npy files.

    Use for:
    - **Embeddings (MAE)**: load_masks=False, returns {'image': [1,H,W]}
    - **Segmentation**: load_masks=True, returns {'image': [1,H,W], 'mask': [C,H,W]}

    Args:
        root: Root directory containing ARRAY_FORMAT/ (and optionally IMAGES/)
        array_dir: Subdir with .npy files (default 'ARRAY_FORMAT')
        images_dir: Subdir with .png fallback (default 'IMAGES'). Used when npy
                    contains dict without 'image' key.
        load_masks: If True, extract masks from npy dict (structure -> mask mapping).
        structure_keys: Keys to treat as mask sources in npy dict. Default: all known.
        transform: Optional transform. If None, uses default resize to target_size.
        target_size: (H, W) for resize. Default (224, 224).
    """

    def __init__(
        self,
        root: str,
        *,
        array_dir: str = "ARRAY_FORMAT",
        images_dir: str = "IMAGES",
        load_masks: bool = False,
        structure_keys: Optional[List[str]] = None,
        transform: Optional[object] = None,
        target_size: Tuple[int, int] = (224, 224),
        preprocessed: bool = False,
    ):
        self.root = Path(root)
        self.array_dir = self.root / array_dir
        self.preprocessed = preprocessed
        if preprocessed:
            self.images_dir = self.root / (images_dir + '_preprocessed')
        else:
            self.images_dir = self.root / images_dir
        self.load_masks = load_masks
        self.structure_keys = structure_keys or STRUCTURE_KEYS
        self.target_size = target_size

        if preprocessed:
            if not self.images_dir.exists():
                raise FileNotFoundError(f"Preprocessed images directory not found: {self.images_dir}")
            self.npy_paths: List[Path] = []
            self.sample_ids: List[str] = []
            for p in sorted(self.images_dir.iterdir()):
                if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    self.npy_paths.append(p)  # repurposed as image_paths in this mode
                    self.sample_ids.append(p.stem)
            if len(self.npy_paths) == 0:
                raise ValueError(f"No PNG/JPG files found in {self.images_dir}")
        else:
            if not self.array_dir.exists():
                raise FileNotFoundError(f"Array directory not found: {self.array_dir}")
            self.npy_paths = sorted(
                p for p in self.array_dir.iterdir() if p.suffix.lower() == ".npy"
            )
            self.sample_ids = [p.stem for p in self.npy_paths]
            if len(self.npy_paths) == 0:
                raise ValueError(f"No .npy files found in {self.array_dir}")

        src = self.images_dir if preprocessed else self.array_dir
        print(f"Loaded {len(self.npy_paths)} samples from {src}")

        self.transform = transform if transform is not None else self._get_default_transform()

    def _get_default_transform(self):
        """Default: resize image (and mask) to target_size."""
        size = self.target_size[0]

        if self.load_masks:

            def transform_fn(sample: Dict) -> Dict:
                img = sample["image"]
                mask = sample["mask"]
                if not isinstance(img, torch.Tensor):
                    img = torch.from_numpy(np.asarray(img, dtype=np.float32))
                if img.dim() == 2:
                    img = img.unsqueeze(0)
                if img.dim() == 3 and img.size(0) > 1:
                    img = img.mean(dim=0, keepdim=True)
                img = resize_keep_aspect_pad(
                    img, size, size, interpolation=T.InterpolationMode.BILINEAR
                )
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
                    img = torch.from_numpy(np.asarray(img, dtype=np.float32))
                if img.dim() == 2:
                    img = img.unsqueeze(0)
                if img.dim() == 3 and img.size(0) > 1:
                    img = img.mean(dim=0, keepdim=True)
                return resize_keep_aspect_pad(
                    img, size, size, interpolation=T.InterpolationMode.BILINEAR
                )

            return resize_img

    def _load_npy(self, path: Path) -> Union[np.ndarray, Dict]:
        """Load .npy file; return array or dict."""
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            item = data.item()
            return item if isinstance(item, dict) else data
        return data

    def _array_to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert numpy array to [1, H, W] tensor, normalize to [0, 1] if needed."""
        t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        if t.dim() == 2:
            t = t.unsqueeze(0)
        elif t.dim() == 3:
            if t.shape[0] in (1, 3) and t.shape[2] != 3:
                pass
            elif t.shape[2] in (1, 3):
                t = t.permute(2, 0, 1)
            if t.size(0) > 1:
                t = t.mean(dim=0, keepdim=True)
        if t.max() > 1.0 and t.min() >= 0:
            t = t / 255.0
        return t

    def __len__(self) -> int:
        return len(self.npy_paths)

    def _getitem_preprocessed(self, idx: int) -> Dict:
        """Load image from preprocessed PNG, masks (if needed) from original NPY."""
        img_path = self.npy_paths[idx]  # actually a PNG path in this mode
        sample_id = self.sample_ids[idx]

        img_pil = Image.open(img_path).convert("L")
        image = T.ToTensor()(img_pil)  # [1, H, W]

        if not self.load_masks:
            image = self.transform(image)
            return {"image": image}

        # Masks still from original NPY
        npy_path = self.array_dir / f"{sample_id}.npy"
        if not npy_path.exists():
            mask = torch.zeros((1, *image.shape[1:]))
        else:
            data = self._load_npy(npy_path)
            if isinstance(data, dict):
                masks = []
                for key in self.structure_keys:
                    if key in data:
                        m = torch.from_numpy(np.asarray(data[key], dtype=np.float32))
                        if m.dim() == 2:
                            m = m.unsqueeze(0)
                        m = (m > 0.5).float() if m.max() <= 1 else (m > 0).float()
                        masks.append(m)
                mask = torch.cat(masks, dim=0) if masks else torch.zeros((1, *image.shape[1:]))
                if mask.shape[1:] != image.shape[1:]:
                    mask = F.interpolate(
                        mask.unsqueeze(0).float(), size=image.shape[1:], mode="nearest"
                    ).squeeze(0)
            else:
                mask = torch.zeros((1, *image.shape[1:]))

        sample = {"image": image, "mask": mask}
        sample = self.transform(sample)
        sample["image_id"] = sample_id
        return sample

    def __getitem__(self, idx: int) -> Dict:
        if self.preprocessed:
            return self._getitem_preprocessed(idx)
        npy_path = self.npy_paths[idx]
        sample_id = self.sample_ids[idx]
        data = self._load_npy(npy_path)

        if isinstance(data, np.ndarray):
            image = self._array_to_tensor(data)
            if not self.load_masks:
                image = self.transform(image)
                return {"image": image}
            mask = torch.zeros((1, image.shape[1], image.shape[2]))
            sample = {"image": image, "mask": mask}
            sample = self.transform(sample)
            sample["image_id"] = sample_id
            return sample
        else:
            # Dict: structure -> mask mapping
            image = None
            if "image" in data:
                image = self._array_to_tensor(data["image"])
            elif self.images_dir.exists():
                png_path = self.images_dir / f"{sample_id}.png"
                if png_path.exists():
                    img_pil = Image.open(png_path).convert("L")
                    image = T.ToTensor()(img_pil)
                else:
                    png_path = self.images_dir / f"{sample_id}.jpg"
                    if png_path.exists():
                        img_pil = Image.open(png_path).convert("L")
                        image = T.ToTensor()(img_pil)
            if image is None:
                raise ValueError(
                    f"No image in npy and no PNG found for {sample_id}. "
                    "Ensure npy has 'image' key or IMAGES/ has matching .png"
                )

            masks = []
            for key in self.structure_keys:
                if key in data:
                    m = torch.from_numpy(np.asarray(data[key], dtype=np.float32))
                    if m.dim() == 2:
                        m = m.unsqueeze(0)
                    m = (m > 0.5).float() if m.max() <= 1 else (m > 0).float()
                    masks.append(m)
            mask = torch.cat(masks, dim=0) if masks else torch.zeros((1, *image.shape[1:]))
            if mask.shape[1:] != image.shape[1:]:
                mask = F.interpolate(
                    mask.unsqueeze(0).float(),
                    size=image.shape[1:],
                    mode="nearest",
                ).squeeze(0)

            if not self.load_masks:
                image = self.transform(image)
                return {"image": image}
            sample = {"image": image, "mask": mask}

        sample = self.transform(sample)
        sample["image_id"] = sample_id
        return sample
