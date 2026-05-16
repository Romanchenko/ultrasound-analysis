
## Acouslic
https://zenodo.org/records/12697994?preview_file=acouslic-ai-train-set.zip

File structure
```
/root
    /images
        /stacked_fetal_ultrasound
            /<UUID_1>.mha
            /<UUID_2>.mha
            ...
    /masks
        /stacked_fetal_abdomen
            /<UUID_1>.mha
            /<UUID_2>.mha
            ...
    /circumferences
        /fetal_abdominal_circumferences_per_sweep.csv
```

For foundation model training we need only /images directory. Each .mha is a set of 840 frames

SimpleITK axes order: (x, y, z/frames) → numpy: (z/frames, y, x)

Let's take around 1/2 of images -- so that in total we have ~13k images. Just taking frames 0, 2, 4, 6, ... of each image into account and use them as individual items through iteration.