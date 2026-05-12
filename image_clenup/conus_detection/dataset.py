"""
Dataset for ultrasound conus detection from CVAT 1.1 XML annotations.

Reads polygon and box annotations, rasterises them to binary masks, and
returns grayscale image + mask pairs.

Dataset structure:
    root/
        for_annotation/         ← .jpg images (annotations_1.xml)
        for_annotation_2/       ← .png images (annotations_2.xml)
        annotations_1.xml
        annotations_2.xml
"""

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# XML parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_cvat_xml(xml_path: Path) -> List[Dict]:
    """Parse CVAT 1.1 XML → list of record dicts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records = []
    for img_el in root.findall("image"):
        shapes = []
        for poly in img_el.findall("polygon"):
            pts = [
                (float(x), float(y))
                for pt in poly.get("points", "").split(";")
                if "," in pt
                for x, y in [pt.split(",")]
            ]
            if len(pts) >= 3:
                shapes.append(("polygon", pts))
        for box in img_el.findall("box"):
            shapes.append((
                "box",
                (float(box.get("xtl")), float(box.get("ytl")),
                 float(box.get("xbr")), float(box.get("ybr"))),
            ))
        records.append({
            "name":   img_el.get("name"),
            "width":  int(img_el.get("width")),
            "height": int(img_el.get("height")),
            "shapes": shapes,
        })
    return records


def _rasterize(shapes: List, width: int, height: int) -> Image.Image:
    """Render annotation shapes into a binary PIL mask image."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for kind, data in shapes:
        if kind == "polygon":
            flat = [coord for pt in data for coord in pt]
            draw.polygon(flat, fill=255)
        elif kind == "box":
            xtl, ytl, xbr, ybr = data
            draw.rectangle([xtl, ytl, xbr, ybr], fill=255)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_ANN = [
    ("annotations_1.xml", "for_annotation"),
    ("annotations_2.xml", "for_annotation_2"),
]


class ConusDataset(Dataset):
    """
    Ultrasound conus segmentation dataset.

    Returns ``{"image": [1, H, W] float32 in [0,1],
                "mask":  [1, H, W] float32 binary}``.

    Args:
        root: Root directory containing the XML files and image folders.
        annotations: List of ``(xml_filename, image_folder)`` pairs, relative to
                     ``root``.  Defaults to the two standard pairs from the readme.
        target_size: Output ``(H, W)`` for both image and mask.
        augment:     If True, apply random augmentations (use for training split).
    """

    def __init__(
        self,
        root: str,
        annotations: Optional[List[Tuple[str, str]]] = None,
        target_size: Tuple[int, int] = (512, 512),
        augment: bool = False,
    ):
        self.root = Path(root)
        self.target_size = target_size
        self.augment = augment

        self.samples: List[Dict] = []
        missing = 0
        for xml_name, _ in (annotations or _DEFAULT_ANN):
            xml_path = self.root / xml_name
            if not xml_path.exists():
                raise FileNotFoundError(f"Annotation file not found: {xml_path}")
            for rec in _parse_cvat_xml(xml_path):
                img_path = self.root / rec["name"]
                if not img_path.exists():
                    missing += 1
                    continue
                self.samples.append({
                    "img_path": img_path,
                    "width":    rec["width"],
                    "height":   rec["height"],
                    "shapes":   rec["shapes"],
                })

        if len(self.samples) == 0:
            raise ValueError(f"No images found under {self.root}")
        if missing:
            print(f"Warning: {missing} annotated images not found on disk — skipped")
        print(f"ConusDataset: {len(self.samples)} samples (augment={augment})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        img  = Image.open(s["img_path"]).convert("L")
        mask = _rasterize(s["shapes"], s["width"], s["height"])

        # Resize to target_size — (W, H) order for PIL
        tw, th = self.target_size[1], self.target_size[0]
        img  = img.resize((tw, th),  Image.BILINEAR)
        mask = mask.resize((tw, th), Image.NEAREST)

        img_t  = TF.to_tensor(img)           # [1, H, W] in [0, 1]
        mask_t = TF.to_tensor(mask)          # [1, H, W] in {0, 1}
        mask_t = (mask_t > 0.5).float()

        if self.augment:
            img_t, mask_t = _augment(img_t, mask_t)

        return {"image": img_t, "mask": mask_t}


def _augment(
    img: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        img  = TF.hflip(img)
        mask = TF.hflip(mask)
    if random.random() < 0.3:
        img  = TF.vflip(img)
        mask = TF.vflip(mask)
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        img  = TF.rotate(img,  angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST,  fill=0)
    if random.random() < 0.5:
        img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
    return img, mask
