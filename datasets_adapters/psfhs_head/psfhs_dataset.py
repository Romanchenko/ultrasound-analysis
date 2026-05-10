"""
Dataset adapter for PSFHS: Pubic Symphysis-Fetal Head Segmentation dataset.

PSFHS is an intrapartum transperineal ultrasound image dataset of 1358 grayscale
images from 1124 pregnant women, originally built for segmentation of the pubic
symphysis (PS) and fetal head (FH) (see PSFHS challenge of MICCAI 2023).

For foundation-model training we only need the raw images; a segmentation mode
is also supported so the same adapter can be reused downstream.

Supports:
- Embeddings training (MAE): returns grayscale images [1, H, W]
- Segmentation validation: returns image + mask with PS and/or FH channels

Dataset structure (see README.md):
    root/
        image_mha/
            03744.mha
            03745.mha
            ...
        label_mha/            (optional, only needed for segmentation)
            03744.mha
            03745.mha
            ...

Label pixel values (in label_mha files):
    0 = background
    1 = pubic symphysis (PS)
    2 = fetal head (FH)

Links:
- https://zenodo.org/records/10969427
- https://www.nature.com/articles/s41597-024-03266-4
- https://github.com/maskoffs/PS-FH-MICCAI23
"""

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


MaskTarget = Literal["ps", "fh", "both"]


def _read_mha(path: Path) -> np.ndarray:
    """Load an ``.mha`` MetaImage file into a 2D numpy array.

    The dataset stores 2D ultrasound frames but MetaImage is inherently
    N-dimensional, so SimpleITK may return an array with a leading singleton
    axis (``[1, H, W]``) which we squeeze out.

    Prefers ``SimpleITK`` (the de-facto tool for ``.mha`` files); falls back
    to ``itk`` if only that is installed.
    """
    try:
        import SimpleITK as sitk  # type: ignore
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)
    except ImportError:
        try:
            import itk  # type: ignore
            img = itk.imread(str(path))
            arr = np.asarray(img)
        except ImportError as e:
            raise ImportError(
                "Reading .mha files requires either 'SimpleITK' or 'itk'. "
                "Install with: pip install SimpleITK"
            ) from e

    arr = np.asarray(arr)
    # Drop singleton dims (SimpleITK wraps 2D frames as [1, H, W]).
    arr = np.squeeze(arr)
    # If the file happens to be RGB (shouldn't for PSFHS, but be safe), average.
    if arr.ndim == 3:
        arr = arr.mean(axis=-1) if arr.shape[-1] in (3, 4) else arr[0]
    if arr.ndim != 2:
        raise ValueError(
            f"Unexpected array shape {arr.shape} from {path}; expected a 2D image."
        )
    return arr


def _array_to_pil_gray(arr: np.ndarray) -> Image.Image:
    """Convert an arbitrary numeric 2D array into a PIL ``'L'`` image.

    For the standard PSFHS release this is a no-op cast (values are already
    ``uint8``), but we min-max-normalise anything else to ``uint8`` so the
    downstream ``ToTensor`` always yields values in ``[0, 1]``.
    """
    if arr.dtype == np.uint8:
        return Image.fromarray(arr, mode="L")

    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="L")


class PSFHSDataset(Dataset):
    """
    PSFHS dataset for intrapartum transperineal ultrasound images.

    Images are returned at their **native** resolution. The MAE pipeline takes
    care of the max-height shrink + per-batch padding
    (:func:`embeddings.vit.train.apply_max_height_shrink`,
    :func:`embeddings.vit.train.mae_pad_collate`), so this dataset performs no
    sizing transforms of its own.

    Use for:
    - **Embeddings (MAE)**: ``load_masks=False``, returns ``{'image': [1,H,W]}``
    - **Segmentation**: ``load_masks=True``, returns
      ``{'image': [1,H,W], 'mask': [C,H,W], 'image_id': str}``

    Args:
        root: Root directory containing ``image_mha/`` (and optionally ``label_mha/``).
        load_masks: If True, load segmentation masks from ``label_mha/``.
        mask_target: ``'ps'``, ``'fh'``, or ``'both'``.
                     ``'both'`` returns mask ``[2,H,W]`` with PS in ch0, FH in ch1.
        transform: Optional callable. If ``None``, the image is returned as a
                   grayscale ``[1, H, W]`` tensor (via ``ToTensor``) with no
                   resizing. When ``load_masks=True``, the transform receives a
                   dict with ``'image'`` and ``'mask'`` and must return a dict
                   with the same keys.
        images_dir: Name of images subdir (default ``'image_mha'``).
        masks_dir: Name of masks subdir (default ``'label_mha'``).
    """

    def __init__(
        self,
        root: str,
        *,
        load_masks: bool = False,
        mask_target: MaskTarget = "both",
        transform: Optional[Callable] = None,
        images_dir: str = "image_mha",
        masks_dir: str = "label_mha",
        target_size: Optional[Any] = None,  # deprecated, ignored
    ):
        if target_size is not None:
            warnings.warn(
                "PSFHSDataset(target_size=...) is deprecated and ignored: "
                "images are now returned at their native resolution and sized by "
                "the MAE pipeline (apply_max_height_shrink + mae_pad_collate).",
                DeprecationWarning, stacklevel=2,
            )

        self.root = Path(root)
        self.load_masks = load_masks
        self.mask_target = mask_target
        self.images_dir_name = images_dir
        self.masks_dir_name = masks_dir

        self.images_dir = self.root / images_dir
        self.masks_dir = self.root / masks_dir

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        self.image_paths: List[Path] = []
        self.image_ids: List[str] = []

        for p in sorted(self.images_dir.iterdir()):
            if p.suffix.lower() == ".mha":
                self.image_paths.append(p)
                self.image_ids.append(p.stem)

        if load_masks and not self.masks_dir.exists():
            raise FileNotFoundError(
                f"Masks directory not found: {self.masks_dir}. "
                "Set load_masks=False for embeddings-only mode."
            )

        if load_masks:
            valid_paths = []
            valid_ids = []
            for path, img_id in zip(self.image_paths, self.image_ids):
                mask_path = self.masks_dir / f"{img_id}.mha"
                if mask_path.exists():
                    valid_paths.append(path)
                    valid_ids.append(img_id)
            self.image_paths = valid_paths
            self.image_ids = valid_ids

        if len(self.image_paths) == 0:
            raise ValueError(f"No valid .mha images found in {self.images_dir}")

        print(f"Loaded {len(self.image_paths)} images from {self.images_dir}")

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

        try:
            arr = _read_mha(image_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load {image_path}: {e}") from e
        image = _array_to_pil_gray(arr)

        if not self.load_masks:
            if callable(self.transform):
                image = self.transform(image)
            else:
                image = T.ToTensor()(image)
                if image.dim() == 2:
                    image = image.unsqueeze(0)
            return {"image": image, "image_id": image_id}

        mask_path = self.masks_dir / f"{image_id}.mha"
        mask_arr = _read_mha(mask_path).astype(np.int64)  # values in {0, 1, 2}

        channels = []
        if self.mask_target in ("ps", "both"):
            channels.append(torch.from_numpy((mask_arr == 1).astype(np.float32)))
        if self.mask_target in ("fh", "both"):
            channels.append(torch.from_numpy((mask_arr == 2).astype(np.float32)))
        mask = torch.stack(channels, dim=0)  # [1,H,W] or [2,H,W]

        sample = {"image": image, "mask": mask}
        sample = self.transform(sample)
        sample["image_id"] = image_id
        return sample
