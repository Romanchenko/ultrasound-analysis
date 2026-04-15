# FOCUS: Four-chamber Ultrasound Image Dataset

Segmentation/detection of thorax and heart in fetal four-chamber ultrasound images.

## Usage

```python
from datasets_adapters.focus import FOCUSDataset

# For MAE embeddings training
ds = FOCUSDataset(root="/path/to/FOCUS-dataset", split="training", load_masks=False)
# Returns: {'image': [1, 224, 224]}

# For segmentation validation
ds = FOCUSDataset(
    root="/path/to/FOCUS-dataset",
    split="validation",
    load_masks=True,
    mask_target="both",  # 'cardiac', 'thorax', or 'both'
)
# Returns: {'image': [1, H, W], 'mask': [2, H, W], 'image_id': str}
# mask: channel 0 = cardiac, channel 1 = thorax
```

## Dataset structure:
```
/root_folder
    /testing
        /annfiles_ellipse
            /001.txt
            /002.txt
        /annfiles_mask
            /001-cardiac.png
            /001-thorax.png
            /002-cardiac.png
            /002-thorax.png
        /annfiles_rectangle
            /001.txt
            /002.txt
        /images
            /001.png
            /002.png
    /training
        /annfiles_ellipse
            /001.txt
            /002.txt
        /annfiles_mask
            /001-cardiac.png
            /001-thorax.png
            /002-cardiac.png
            /002-thorax.png
        /annfiles_rectangle
            /001.txt
            /002.txt
        /images
            /001.png
            /002.png
    /validation
        /annfiles_ellipse
            /001.txt
            /002.txt
        /annfiles_mask
            /001-cardiac.png
            /001-thorax.png
            /002-cardiac.png
            /002-thorax.png
        /annfiles_rectangle
            /001.txt
            /002.txt
        /images
            /001.png
            /002.png
```

annfiles_rectangle 001.txt sample:
```
289.3 544.4 596.6 416.5 509.1 206.3 201.9 334.2 cardiac 0
483.0 759.0 895.0 382.8 518.5 -29.5 106.5 346.6 thorax 0
```

annfiles_ellipse 001.txt sample:
```
399.2 375.4 166.4 113.8 157.4 cardiac
500.7 364.7 279.0 279.2 137.6 thorax
```


Links:
https://github.com/szuboy/FOCUS-dataset/tree/main
https://zenodo.org/records/14597550
https://zenodo.org/records/14597550/files/FOCUS-dataset.zip
