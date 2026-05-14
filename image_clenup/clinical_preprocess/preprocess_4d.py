"""
Extract the first frame from 4-D DICOM arrays in the clinical dataset.

Input structure:
    root/studies/{STUDY}.zip  →  internal: {path}/{file}.dcm

Output structure (mirrors ZIP-internal paths):
    root/studies_first_frame/{STUDY}/{path}/{file}.png

Steps per ZIP:
  1. Extract the ZIP directly into studies_first_frame/{STUDY}/
  2. Walk every extracted .dcm, read pixel_array
  3. If ndim == 4: take arr[0]
  4. Save as .png alongside (then remove the .dcm)

Only action applied to pixel data: if pixel_array.ndim == 4, take arr[0].
No normalisation or other transforms.
"""

import zipfile
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


def preprocess(
    root: str,
    studies_subdir: str = "studies",
    output_subdir: str = "studies_first_frame",
    skip_existing: bool = True,
) -> None:
    root = Path(root)
    studies_dir = root / studies_subdir
    output_dir = root / output_subdir

    zip_paths = sorted(studies_dir.glob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No ZIP files found in {studies_dir}")

    n_saved = n_skipped = n_error = 0

    for zip_path in zip_paths:
        zip_out = output_dir / zip_path.stem
        print(f"Extracting {zip_path.name} → {zip_out} ...")

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(zip_out)
        except zipfile.BadZipFile:
            print(f"  WARNING: bad ZIP, skipping {zip_path.name}")
            continue

        for dcm_path in sorted(zip_out.rglob("*.dcm")):
            png_path = dcm_path.with_suffix(".png")

            if skip_existing and png_path.exists():
                dcm_path.unlink()
                n_skipped += 1
                continue

            try:
                ds = pydicom.dcmread(str(dcm_path), force=True)
                arr = ds.pixel_array

                if arr.ndim == 4:
                    arr = arr[0]

                _save_png(arr, png_path)
                dcm_path.unlink()
                n_saved += 1
            except Exception as exc:
                print(f"  WARNING: {dcm_path.relative_to(output_dir)}: {exc}")
                n_error += 1

    print(f"\nDone.  saved={n_saved}  skipped={n_skipped}  errors={n_error}")
    print(f"Output: {output_dir}")


def _save_png(arr: np.ndarray, path: Path) -> None:
    if arr.dtype == np.uint8:
        img = Image.fromarray(arr)
    elif arr.dtype == np.uint16:
        img = Image.fromarray(arr.astype(np.int32), mode="I")
    else:
        # Normalise to uint8 only to satisfy PNG requirements for other dtypes
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr8 = ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            arr8 = np.zeros_like(arr, dtype=np.uint8)
        img = Image.fromarray(arr8)
    img.save(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Dataset root directory")
    parser.add_argument("--studies-subdir", default="studies")
    parser.add_argument("--output-subdir", default="studies_first_frame")
    parser.add_argument("--no-skip", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    preprocess(
        args.root,
        studies_subdir=args.studies_subdir,
        output_subdir=args.output_subdir,
        skip_existing=args.skip_existing,
    )
