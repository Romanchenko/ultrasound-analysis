"""
CLI entry point for the preprocessing pipeline.

SOURCE_MAPPING accepts JSON:
    SOURCE_MAPPING='{"src1": "dst1", "src2": "dst2"}'

Or the legacy shorthand (single source → single dest):
    SOURCE_DIR=/data/image_mha  DEST_DIR=/data/image_mha_preprocessed

Usage examples:
    SOURCE_MAPPING='{"data/image_mha": "data/image_mha_preprocessed"}' \\
    CONUS_CHECKPOINT=/ckpts/best_conus.pt \\
    python image_clenup/run_pipeline.py

    # Multiple dirs in one run:
    SOURCE_MAPPING='{
        "/data/image_mha": "/data/image_mha_preprocessed",
        "/data/Images": "/data/Images_preprocessed"
    }' python image_clenup/run_pipeline.py
"""

import json
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

# ── Source mapping ─────────────────────────────────────────────────────────────
_mapping_json = os.environ.get("SOURCE_MAPPING", "").strip()
if _mapping_json:
    try:
        SOURCE_MAPPING = json.loads(_mapping_json)
    except json.JSONDecodeError as e:
        print(f"Error: SOURCE_MAPPING is not valid JSON: {e}")
        sys.exit(1)
else:
    # Legacy single-pair shorthand
    _src = os.environ.get("SOURCE_DIR", "").strip()
    _dst = os.environ.get("DEST_DIR", "").strip()
    if not _src or not _dst:
        print(
            "Error: set SOURCE_MAPPING (JSON dict) or SOURCE_DIR + DEST_DIR env vars.\n"
            "Example:\n"
            "  SOURCE_MAPPING='{\"src\": \"dst\"}' python image_clenup/run_pipeline.py\n"
            "  SOURCE_DIR=/data/imgs DEST_DIR=/data/imgs_preprocessed python image_clenup/run_pipeline.py"
        )
        sys.exit(1)
    SOURCE_MAPPING = {_src: _dst}

CONUS_CHECKPOINT = os.environ.get("CONUS_CHECKPOINT", "").strip() or None
WORKERS          = int(os.environ.get("WORKERS", 4))
BATCH_SIZE_GPU   = int(os.environ.get("BATCH_SIZE_GPU", 16))
SKIP_EXISTING    = bool(int(os.environ.get("SKIP_EXISTING", 1)))
DEVICE           = os.environ.get("DEVICE", "").strip() or None

manifest_paths = run_pipeline(
    source_mapping=SOURCE_MAPPING,
    conus_checkpoint=CONUS_CHECKPOINT,
    workers=WORKERS,
    batch_size_gpu=BATCH_SIZE_GPU,
    skip_existing=SKIP_EXISTING,
    device=DEVICE,
)

for dst, mp in manifest_paths.items():
    print(f"Manifest: {mp}")
