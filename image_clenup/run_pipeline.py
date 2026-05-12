"""
CLI entry point for the preprocessing pipeline.

Usage:
    SOURCE_PATHS=/data/fpdb,/data/clinical \
    OUTPUT_DIR=/data/preprocessed \
    CONUS_CHECKPOINT=/ckpts/best_conus.pt \
    python image_clenup/run_pipeline.py

All configuration via environment variables (see pipeline.py for full list).
"""

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from image_clenup.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

SOURCE_PATHS      = os.environ.get("SOURCE_PATHS", "").strip()
OUTPUT_DIR        = os.environ.get("OUTPUT_DIR", "").strip()
CONUS_CHECKPOINT  = os.environ.get("CONUS_CHECKPOINT", "").strip() or None
WORKERS           = int(os.environ.get("WORKERS", 4))
BATCH_SIZE_GPU    = int(os.environ.get("BATCH_SIZE_GPU", 16))
SKIP_EXISTING     = int(os.environ.get("SKIP_EXISTING", 1))
DEVICE            = os.environ.get("DEVICE", "").strip() or None

if not SOURCE_PATHS:
    print("Error: SOURCE_PATHS must be set (comma-separated list of dirs/files)")
    sys.exit(1)
if not OUTPUT_DIR:
    print("Error: OUTPUT_DIR must be set")
    sys.exit(1)

run_pipeline(
    source_paths=SOURCE_PATHS.split(","),
    output_dir=OUTPUT_DIR,
    conus_checkpoint=CONUS_CHECKPOINT,
    workers=WORKERS,
    batch_size_gpu=BATCH_SIZE_GPU,
    skip_existing=bool(SKIP_EXISTING),
    device=DEVICE,
)
