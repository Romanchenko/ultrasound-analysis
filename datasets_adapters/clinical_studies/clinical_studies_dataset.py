"""
Dataset adapter for the Clinical Ultrasound Image Repository.

Iterates over DICOM files inside ZIP archives without extracting them.
Returns grayscale images as {'image': [1, H, W] float32 tensor in [0, 1]}.
Labels/patient metadata are ignored.

Dataset structure (raw):
    root_dir/
        studies/
            {STUDY_NAME}.zip   (one per study)
                /{study_folder}/images/dicom/*.dcm

Dataset structure (preprocessed):
    root_dir/
        studies_preprocessed/
            {stem}.png   (flat directory of PNGs from the pipeline)
"""

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class ClinicalStudiesDataset(Dataset):
    """
    PyTorch Dataset over DICOM files stored inside ZIP archives, or preprocessed PNGs.

    Raw mode: indexes DICOMs inside ZIPs; each __getitem__ opens the ZIP on demand.
    Preprocessed mode (``preprocessed=True``): reads flat PNG/JPG files from
    ``{studies_subdir}_preprocessed/`` sibling directory produced by the pipeline.

    Args:
        root: Root directory.
        studies_subdir: Name of the subdir holding ZIPs (default 'studies').
        preprocessed: If True, load from ``{studies_subdir}_preprocessed/`` as PNG/JPG.
    """

    def __init__(
        self,
        root: str,
        *,
        studies_subdir: str = "studies",
        preprocessed: bool = False,
    ):
        self.root = Path(root)
        self.preprocessed = preprocessed

        if preprocessed:
            self.prep_dir = self.root / (studies_subdir + '_preprocessed')
            if not self.prep_dir.exists():
                raise FileNotFoundError(
                    f"Preprocessed directory not found: {self.prep_dir}"
                )
            self._prep_paths: List[Path] = sorted(
                p for p in self.prep_dir.iterdir()
                if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
            )
            if len(self._prep_paths) == 0:
                raise ValueError(f"No PNG/JPG files found in {self.prep_dir}")
            print(f"ClinicalStudiesDataset: {len(self._prep_paths)} preprocessed images from {self.prep_dir}")
        else:
            import pydicom  # noqa: F401  # ensure available before indexing
            self.studies_dir = self.root / studies_subdir
            if not self.studies_dir.exists():
                raise FileNotFoundError(f"Studies directory not found: {self.studies_dir}")
            # Index: list of (zip_path, member_path_inside_zip)
            self._index: List[Tuple[Path, str]] = []
            self._scan_zips()
            if len(self._index) == 0:
                raise ValueError(f"No DICOM files found under {self.studies_dir}")
            print(
                f"ClinicalStudiesDataset: {len(self._index)} DICOM files across "
                f"{len(list(self.studies_dir.glob('*.zip')))} ZIP archives"
            )

    def _scan_zips(self) -> None:
        for zip_path in sorted(self.studies_dir.glob("*.zip")):
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    for member in zf.namelist():
                        if member.lower().endswith(".dcm"):
                            self._index.append((zip_path, member))
            except zipfile.BadZipFile:
                print(f"Warning: skipping bad ZIP {zip_path.name}")

    def __len__(self) -> int:
        if self.preprocessed:
            return len(self._prep_paths)
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.preprocessed:
            img = Image.open(self._prep_paths[idx]).convert("L")
            import torchvision.transforms as T
            return {"image": T.ToTensor()(img)}

        import pydicom
        zip_path, member = self._index[idx]
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(member) as f:
                data = f.read()

        ds = pydicom.dcmread(io.BytesIO(data), force=True)
        arr = ds.pixel_array  # [H, W], [N, H, W], or [N, H, W, C]

        # Color video (N, H, W, C) → take first frame → (H, W, C)
        if arr.ndim == 4:
            arr = arr[0]

        # (H, W, C) color image → average channels to grayscale
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            arr = arr.mean(axis=-1)
        elif arr.ndim == 3:
            # (N, H, W) multi-frame cine → pick middle frame
            arr = arr[arr.shape[0] // 2]

        arr = arr.astype(np.float32)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = np.zeros_like(arr)

        return {"image": torch.from_numpy(arr).unsqueeze(0)}
