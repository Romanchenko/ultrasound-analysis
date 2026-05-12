## Cleanup everything but the US conus

Regular ultrsound usually has some texts and technical info outside of the conus representing the research object. Current model should detect this conus. Later we can apply it by removing everything but the conus from the image.

Out hypothesys is that it will help in learning useful features from the image itself.

### Dataset

Dataset is collected and annotated manually from several other datasets.

Dataset structure:
```
/DATASET_DIR
    /for_annotation
        /<IMAGE_NAME_1>.jpg
        ...
    /for_annotation_2
        /<IMAGE_NAME_N>.jpg
        ...
    annotations_1.xml
    annotations_2.xml
```

Annotations XML is labels export from CVAT in "CVAT for images 1.1" format.

You can find those xmls for reference in data/sample folder of this repository.

`annotations-1.xml` if for `for_annotation` folder and `annotations_2.xml` is for `for_annotation_2`.