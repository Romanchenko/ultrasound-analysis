# Fetal Abdominal Structures Segmentation Dataset

Source: [Mendeley Data 4gcpm9dsc3](https://data.mendeley.com/datasets/4gcpm9dsc3/1)

~1500 fetal abdomen circumference ultrasound images with segmentations for: abdominal aorta, intrahepatic umbilical vein, stomach, liver.

## Usage

```python
from datasets_adapters.abdominal_segmentation import AbdominalSegmentationDataset

# For MAE embeddings training (loads from .npy)
ds = AbdominalSegmentationDataset(root="/path/to/dataset-root", load_masks=False)
# Returns: {'image': [1, 224, 224]}

# For segmentation validation
ds = AbdominalSegmentationDataset(
    root="/path/to/dataset-root",
    load_masks=True,
)
# Returns: {'image': [1, H, W], 'mask': [4, H, W], 'image_id': str}
# mask channels: abdominal_aorta, intrahepatic_umbilical_vein, stomach, liver
```

Each .npy file may contain a raw image array or a dict mapping structure names to masks. If the dict lacks an 'image' key, the PNG from IMAGES/ is loaded as fallback.

## Dataset structure:

```
/dataset-root
    /ARRAY_FORMAT
        /P01_IMG1.npy
        /P01_IMG2.npy
        /P02_IMG1.npy
    /IMAGES
        /P01_IMG1.png
        /P01_IMG2.png
        /P02_IMG1.png
```