# PSFHS — Pubic Symphysis & Fetal Head Segmentation

Intrapartum transperineal ultrasound image dataset (1358 grayscale images from 1124
pregnant women) originally built for segmentation of the pubic symphysis (PS) and
fetal head (FH). For foundation-model training we use only `image_mha/`.

- Paper: <https://www.nature.com/articles/s41597-024-03266-4>
- Data (Zenodo): <https://doi.org/10.5281/zenodo.10969427>
- Challenge code / baselines: <https://github.com/maskoffs/PS-FH-MICCAI23>

## Dataset structure

```
/root_folder
    /image_mha
        /03744.mha
        /03745.mha
        ...
    /label_mha           # optional, only needed for segmentation
        /03744.mha
        /03745.mha
        ...
```

Label pixel values (`label_mha/`): `0`=background, `1`=pubic symphysis, `2`=fetal head.

## Dependency

`.mha` (MetaImage) files require `SimpleITK` (preferred) or `itk`:

```bash
pip install SimpleITK
```

## Usage

```python
from datasets_adapters.psfhs_head import PSFHSDataset

ds = PSFHSDataset(root="/path/to/PSFHS")
sample = ds[0]
sample["image"].shape  # torch.Size([1, H, W]), values in [0, 1]
```

For segmentation evaluation:

```python
ds = PSFHSDataset(root="/path/to/PSFHS", load_masks=True, mask_target="both")
s = ds[0]
s["image"].shape  # [1, H, W]
s["mask"].shape   # [2, H, W] — channel 0: PS, channel 1: FH
```
