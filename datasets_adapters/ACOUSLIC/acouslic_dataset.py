"""
Dataset adapter for ACOUSLIC fetal ultrasound sweep dataset.

Each .mha sweep volume contains ~840 stacked cine frames. Items are individual
frames, subsampled via ``frame_stride`` to reduce temporal redundancy.

NOTE: Each __getitem__ call loads the full .mha volume to extract one frame.
      For large-scale training, pre-extracting frames to individual PNGs is
      strongly recommended to avoid repeated volume reads.

Dataset structure:
    root/
        images/
            stacked_fetal_ultrasound/
                <UUID_1>.mha
                <UUID_2>.mha
                ...
        masks/
            stacked_fetal_abdomen/
                <UUID_1>.mha
                ...
        circumferences/
            fetal_abdominal_circumferences_per_sweep.csv

SimpleITK axis order: (x, y, z/frames) → numpy shape: (frames, H, W).

Source: https://zenodo.org/records/12697994
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ACOUSLICDataset(Dataset):
    """
    ACOUSLIC stacked fetal ultrasound sweep dataset.

    Each .mha file contains ~840 cine frames. Items are individual frames
    identified by ``(uuid, frame_idx)`` pairs built at init by reading each
    volume's header (pixel data is NOT loaded at init).

    Use for:
    - **Embeddings (MAE)**: ``load_masks=False``, returns ``{'image': [1,H,W]}``
    - **Segmentation**: ``load_masks=True``, returns ``{'image': [1,H,W], 'mask': [1,H,W]}``

    Args:
        root: Root directory containing ``images/`` and optionally ``masks/``.
        frame_stride: Step between frames (default 2 → every other frame,
                      ~420 items per sweep).
        load_masks: If True, load paired abdomen segmentation masks.
        transform: Optional callable. If ``None``, images are returned as
                   ``[1, H, W]`` float tensors in [0, 1] at native resolution.
                   When ``load_masks=True``, receives and returns a dict with
                   ``'image'`` and ``'mask'`` keys.
        uuids: Optional list of UUID strings to restrict to (for splits).
               Filenames are ``<uuid>.mha``.
    """

    _IMAGES_SUBDIR = "images/stacked_fetal_ultrasound"
    _MASKS_SUBDIR = "masks/stacked_fetal_abdomen"

    def __init__(
        self,
        root: str,
        *,
        frame_stride: int = 2,
        load_masks: bool = False,
        transform: Optional[Callable] = None,
        uuids: Optional[List[str]] = None,
    ):
        import SimpleITK as sitk  # deferred: not always installed

        self.root = Path(root)
        self.frame_stride = frame_stride
        self.load_masks = load_masks

        images_dir = self.root / self._IMAGES_SUBDIR
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        if uuids is not None:
            mha_paths = [images_dir / f"{uid}.mha" for uid in uuids]
            missing = [p for p in mha_paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} .mha file(s) not found, e.g. {missing[0]}"
                )
        else:
            mha_paths = sorted(images_dir.glob("*.mha"))

        if not mha_paths:
            raise ValueError(f"No .mha files found under {images_dir}")

        if load_masks:
            masks_dir = self.root / self._MASKS_SUBDIR
            if not masks_dir.exists():
                raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
            self.masks_dir: Optional[Path] = masks_dir
        else:
            self.masks_dir = None

        # Build flat (mha_path, frame_idx) index.
        # Read headers only (fast — no pixel data transferred).
        reader = sitk.ImageFileReader()
        self.index: List[Tuple[Path, int]] = []
        for mha_path in mha_paths:
            reader.SetFileName(str(mha_path))
            reader.ReadImageInformation()
            n_frames = reader.GetSize()[2]  # ITK Size: (x, y, z=frames)
            for fi in range(0, n_frames, frame_stride):
                self.index.append((mha_path, fi))

        print(
            f"Loaded {len(self.index)} frames ({frame_stride=}) "
            f"from {len(mha_paths)} sweep(s) in {images_dir}"
        )

        self.transform = transform if transform is not None else self._default_transform

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    @property
    def _default_transform(self) -> Callable:
        if self.load_masks:
            def _fn(sample: Dict) -> Dict:
                return {"image": self._to_tensor(sample["image"]), "mask": sample["mask"]}
            return _fn
        return self._to_tensor

    @staticmethod
    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        """Numpy (H, W) uint8/float → [1, H, W] float tensor in [0, 1]."""
        t = torch.from_numpy(arr.astype(np.float32))
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if t.max() > 1.0:
            t = t / 255.0
        return t

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        import SimpleITK as sitk

        mha_path, frame_idx = self.index[idx]
        uuid = mha_path.stem

        volume = sitk.GetArrayFromImage(sitk.ReadImage(str(mha_path)))
        # volume shape after GetArrayFromImage: (frames, H, W)
        frame = volume[frame_idx]  # (H, W)

        if not self.load_masks:
            image = self.transform(frame)
            return {"image": image, "image_id": f"{uuid}/{frame_idx:04d}"}

        # Load paired mask volume
        mask_path = self.masks_dir / mha_path.name
        if mask_path.exists():
            mask_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
            mask_frame = mask_vol[frame_idx]  # (H, W)
            mask = torch.from_numpy((mask_frame > 0).astype(np.float32)).unsqueeze(0)
        else:
            mask = torch.zeros((1, frame.shape[0], frame.shape[1]), dtype=torch.float32)

        sample = self.transform({"image": frame, "mask": mask})
        sample["image_id"] = f"{uuid}/{frame_idx:04d}"
        return sample
