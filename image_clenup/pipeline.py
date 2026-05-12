"""
Batch preprocessing pipeline for fetal ultrasound images.

For each source image:
  1. Load as grayscale (JPG/PNG/DCM/DICOM-in-ZIP/MHA/NPY)
  2. Normalise: clip to [1st, 99th] percentile, then scale to [0, 1]
  3. Apply conus mask (zero non-US region) — skipped if no checkpoint given
  4. Save as 8-bit PNG at native resolution to the mapped output directory
  5. Write/append manifest.csv (one per output directory)

Images are NOT resized — native resolution is preserved.
Resizing is the model's responsibility at training time.

Accepts a source→destination mapping so each input folder writes its
preprocessed PNGs to a dedicated sibling folder.  Designed to be idempotent:
files already present in a manifest are skipped when skip_existing=True.
"""

import csv
import io
import logging
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Supported extensions
# ─────────────────────────────────────────────────────────────────────────────

_RASTER_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_DCM_EXTS    = {".dcm", ".dicom"}
_MHA_EXTS    = {".mha", ".mhd"}
_NPY_EXTS    = {".npy"}
_ZIP_EXTS    = {".zip"}

_ALL_EXTS = _RASTER_EXTS | _DCM_EXTS | _MHA_EXTS | _NPY_EXTS | _ZIP_EXTS


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def _discover_files(paths: List[str]) -> List[Path]:
    """Recursively walk dirs; return all paths with a supported extension."""
    found = []
    for p in paths:
        p = Path(p.strip())
        if p.is_file():
            if p.suffix.lower() in _ALL_EXTS:
                found.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in _ALL_EXTS:
                    found.append(f)
        else:
            logger.warning("Path not found: %s", p)
    return found


def _dst_path(src: Path, output_dir: Path) -> Path:
    return output_dir / f"{src.stem}.png"


# ─────────────────────────────────────────────────────────────────────────────
# Image loaders  →  float32 HxW in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def _load_raster(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _load_dcm(data: bytes) -> np.ndarray:
    import pydicom
    ds  = pydicom.dcmread(io.BytesIO(data), force=True)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        arr = arr.mean(axis=-1)
    elif arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-6)


def _load_bare_dcm(path: Path) -> np.ndarray:
    return _load_dcm(path.read_bytes())


def _load_zip_dcms(path: Path) -> List[Tuple[np.ndarray, str]]:
    """Return list of (array, member_name) for all DCM files inside a ZIP."""
    results = []
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for member in zf.namelist():
                if member.lower().endswith(".dcm"):
                    try:
                        data = zf.read(member)
                        results.append((_load_dcm(data), member))
                    except Exception as exc:
                        logger.debug("Skipping %s/%s: %s", path.name, member, exc)
    except zipfile.BadZipFile:
        logger.warning("Bad ZIP: %s", path)
    return results


def _load_mha(path: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError("SimpleITK required for MHA files: pip install SimpleITK")
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    if img.ndim == 3:
        img = img[img.shape[0] // 2]
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-6)


def _load_npy(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        item = data.item()
        if isinstance(item, dict):
            data = item.get("image", next(iter(item.values())))
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4):
            arr = arr.mean(axis=-1)
        else:
            arr = arr[arr.shape[0] // 2]
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Image transforms
# ─────────────────────────────────────────────────────────────────────────────

def _letterbox(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize longest side to `size`, pad shorter side with zeros (centre)."""
    h, w = arr.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    img  = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    img  = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = np.zeros((size, size), dtype=np.float32)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = np.asarray(img, dtype=np.float32) / 255.0
    return canvas


def _percentile_norm(arr: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Clip to [p_lo, p_hi] percentile then min-max to [0, 1]."""
    lo = float(np.percentile(arr, p_lo))
    hi = float(np.percentile(arr, p_hi))
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _to_png_bytes(arr: np.ndarray) -> bytes:
    img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Single-file processing (returns list because ZIPs → multiple outputs)
# ─────────────────────────────────────────────────────────────────────────────

def _process_one(
    src: Path,
    output_dir: Path,
    skip_existing: bool,
) -> List[Dict]:
    """
    Load and normalise one source file at native resolution.
    Returns a list of manifest row dicts (one per DCM inside a ZIP).
    Does NOT apply the conus mask (done in GPU batches later).
    """
    ext = src.suffix.lower()
    rows = []

    def _make_row(arr, dst, orig_h, orig_w, fmt, status="ok"):
        return {
            "src_path": str(src),
            "dst_path": str(dst),
            "format":   fmt,
            "orig_h":   orig_h,
            "orig_w":   orig_w,
            "conus_coverage_pct": "",   # filled later
            "status":   status,
            "_arr":     arr,            # temp; removed before CSV write
        }

    try:
        if ext in _ZIP_EXTS:
            dcm_list = _load_zip_dcms(src)
            for raw, member in dcm_list:
                orig_h, orig_w = raw.shape
                dst = output_dir / f"{Path(member).stem}.png"
                if skip_existing and dst.exists():
                    continue
                arr = _percentile_norm(raw)
                rows.append(_make_row(arr, dst, orig_h, orig_w, "dcm_zip"))

        elif ext in _DCM_EXTS:
            raw = _load_bare_dcm(src)
            orig_h, orig_w = raw.shape
            dst = _dst_path(src, output_dir)
            if not (skip_existing and dst.exists()):
                arr = _percentile_norm(raw)
                rows.append(_make_row(arr, dst, orig_h, orig_w, "dcm"))

        elif ext in _MHA_EXTS:
            raw = _load_mha(src)
            orig_h, orig_w = raw.shape
            dst = _dst_path(src, output_dir)
            if not (skip_existing and dst.exists()):
                arr = _percentile_norm(raw)
                rows.append(_make_row(arr, dst, orig_h, orig_w, "mha"))

        elif ext in _NPY_EXTS:
            raw = _load_npy(src)
            orig_h, orig_w = raw.shape
            dst = _dst_path(src, output_dir)
            if not (skip_existing and dst.exists()):
                arr = _percentile_norm(raw)
                rows.append(_make_row(arr, dst, orig_h, orig_w, "npy"))

        elif ext in _RASTER_EXTS:
            raw = _load_raster(src)
            orig_h, orig_w = raw.shape
            dst = _dst_path(src, output_dir)
            if not (skip_existing and dst.exists()):
                arr = _percentile_norm(raw)
                rows.append(_make_row(arr, dst, orig_h, orig_w, ext.lstrip(".")))

    except Exception as exc:
        logger.warning("Error loading %s: %s", src, exc)
        rows.append({
            "src_path": str(src), "dst_path": "", "format": ext.lstrip("."),
            "orig_h": "", "orig_w": "", "conus_coverage_pct": "", "status": "load_error",
            "_arr": None,
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Conus masking (GPU-batched)
# ─────────────────────────────────────────────────────────────────────────────

_CONUS_INFER_SIZE = 512  # model's training resolution


def _apply_conus_batch(
    model,
    arrs: List[np.ndarray],
    device,
    threshold: float = 0.5,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Run ConusUNet on a batch of variable-size HxW float32 arrays.

    Each image is letterboxed to _CONUS_INFER_SIZE for inference;
    the resulting mask is bilinearly upsampled back to the native resolution
    before being applied.  Returns (masked_arrs, coverage_pcts).
    """
    import torch
    import torch.nn.functional as F

    native_shapes = [a.shape for a in arrs]

    # Resize all to inference size for batched forward pass
    resized = []
    for a in arrs:
        pil = Image.fromarray((a * 255).clip(0, 255).astype(np.uint8))
        pil = pil.resize((_CONUS_INFER_SIZE, _CONUS_INFER_SIZE), Image.BILINEAR)
        resized.append(np.asarray(pil, dtype=np.float32) / 255.0)

    imgs = torch.from_numpy(np.stack(resized)).unsqueeze(1).float().to(device)  # [B,1,S,S]
    masks_infer = model.predict(imgs, threshold=threshold).cpu()                # [B,1,S,S]

    out_arrs, coverages = [], []
    for arr, mask_t, (orig_h, orig_w) in zip(arrs, masks_infer, native_shapes):
        # Upsample mask back to native resolution
        mask_native = F.interpolate(
            mask_t.unsqueeze(0), size=(orig_h, orig_w), mode="nearest"
        ).squeeze().numpy()                             # [H, W] binary float
        out_arrs.append(arr * mask_native)
        coverages.append(float(mask_native.mean() * 100.0))

    return out_arrs, coverages


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

_MANIFEST_COLS = [
    "src_path", "dst_path", "format", "orig_h", "orig_w",
    "conus_coverage_pct", "status",
]


def run_pipeline(
    source_mapping: Dict[str, str],
    conus_checkpoint: Optional[str] = None,
    workers: int = 4,
    batch_size_gpu: int = 16,
    skip_existing: bool = True,
    device=None,
    conus_threshold: float = 0.5,
) -> Dict[str, str]:
    """
    Run the preprocessing pipeline for a source→destination mapping.

    Args:
        source_mapping: dict mapping each source directory (or file) to its
                        output directory, e.g.
                        ``{"/data/image_mha": "/data/image_mha_preprocessed"}``.
        conus_checkpoint: path to a ConusUNet checkpoint, or None to skip masking.
        workers: number of I/O threads for parallel loading.
        batch_size_gpu: images per GPU forward pass for the conus model.
        skip_existing: skip files already recorded as ``ok`` in the manifest.
        device: torch device; auto-detected when None.
        conus_threshold: binary threshold for conus mask.

    Returns:
        Dict mapping each output directory to the path of its ``manifest.csv``.
    """
    import torch

    mapping: Dict[Path, Path] = {
        Path(s.strip()): Path(d.strip()) for s, d in source_mapping.items()
    }

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load conus model once
    conus_model = None
    if conus_checkpoint:
        conus_ckpt = Path(conus_checkpoint)
        if not conus_ckpt.exists():
            logger.warning("CONUS_CHECKPOINT not found: %s — masking disabled", conus_ckpt)
        else:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from image_clenup.conus_detection.model import ConusUNet
            from image_clenup.conus_detection.train import load_checkpoint
            conus_model = ConusUNet().to(device)
            load_checkpoint(str(conus_ckpt), conus_model, optimizer=None)
            conus_model.eval()
            logger.info("Loaded conus model from %s", conus_ckpt)

    # Per-output-dir: create dirs, open CSV writers, read existing dsts
    existing_dsts: set = set()
    writers:   Dict[Path, csv.DictWriter] = {}
    csv_files: Dict[Path, object] = {}
    manifest_paths: Dict[str, str] = {}

    for dst_dir in mapping.values():
        dst_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = dst_dir / "manifest.csv"
        manifest_paths[str(dst_dir)] = str(manifest_path)

        if skip_existing and manifest_path.exists():
            with open(manifest_path, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("status") == "ok":
                        existing_dsts.add(row["dst_path"])

        csv_existed = manifest_path.exists()
        fh = open(manifest_path, "a", newline="")
        w = csv.DictWriter(fh, fieldnames=_MANIFEST_COLS)
        if not csv_existed:
            w.writeheader()
        writers[dst_dir] = w
        csv_files[dst_dir] = fh

    # Discover all (src, dst_dir) pairs
    all_tasks: List[Tuple[Path, Path]] = []
    for src_dir, dst_dir in mapping.items():
        for f in _discover_files([str(src_dir)]):
            all_tasks.append((f, dst_dir))
    logger.info("Discovered %d source files across %d source dirs", len(all_tasks), len(mapping))

    # Process files in parallel (I/O) → collect rows with arrays
    pending_rows: List[Dict] = []
    n_skipped = n_error = n_queued = 0

    def _process_and_collect(src: Path, dst_dir: Path) -> List[Dict]:
        rows = _process_one(src, dst_dir, skip_existing=False)
        for r in rows:
            r["_dst_dir"] = dst_dir
        return rows

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_and_collect, src, dst_dir): None
                   for src, dst_dir in all_tasks}
        for fut in as_completed(futures):
            for row in fut.result():
                dst = row["dst_path"]
                writer = writers[row["_dst_dir"]]
                fh = csv_files[row["_dst_dir"]]
                if row["status"] == "load_error":
                    n_error += 1
                    _write_row(writer, row)
                    fh.flush()
                elif skip_existing and dst in existing_dsts:
                    n_skipped += 1
                elif row.get("_arr") is None:
                    n_skipped += 1
                else:
                    pending_rows.append(row)
                    n_queued += 1

    logger.info("Queued %d | Skipped %d | Load errors %d", n_queued, n_skipped, n_error)

    # GPU-batched conus masking + save
    n_ok = 0
    for batch_start in range(0, len(pending_rows), batch_size_gpu):
        batch = pending_rows[batch_start: batch_start + batch_size_gpu]
        arrs = [r["_arr"] for r in batch]

        if conus_model is not None:
            try:
                arrs, coverages = _apply_conus_batch(conus_model, arrs, device, conus_threshold)
            except Exception as exc:
                logger.warning("Conus batch error: %s", exc)
                coverages = [""] * len(batch)
        else:
            coverages = [""] * len(batch)

        for row, arr, cov in zip(batch, arrs, coverages):
            dst = Path(row["dst_path"])
            writer = writers[row["_dst_dir"]]
            fh = csv_files[row["_dst_dir"]]
            try:
                dst.write_bytes(_to_png_bytes(arr))
                row["conus_coverage_pct"] = f"{cov:.1f}" if cov != "" else ""
                row["status"] = "ok"
                n_ok += 1
            except Exception as exc:
                logger.warning("Save error %s: %s", dst, exc)
                row["status"] = "save_error"
            _write_row(writer, row)
        for fh in csv_files.values():
            fh.flush()

        done = min(batch_start + batch_size_gpu, len(pending_rows))
        print(f"\r  Processed {done}/{len(pending_rows)}...", end="", flush=True)

    print()
    for fh in csv_files.values():
        fh.close()

    total_out = len(mapping)
    logger.info("Done: %d saved, %d skipped, %d errors across %d output dir(s)",
                n_ok, n_skipped, n_error, total_out)
    for dst_dir, mpath in manifest_paths.items():
        print(f"  {dst_dir}  →  {mpath}")
    return manifest_paths


def _write_row(writer: csv.DictWriter, row: Dict) -> None:
    """Write a manifest row, stripping the internal _arr key."""
    clean = {k: row.get(k, "") for k in _MANIFEST_COLS}
    writer.writerow(clean)
