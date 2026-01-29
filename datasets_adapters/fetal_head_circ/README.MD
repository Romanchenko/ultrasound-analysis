# Automated measurement of fetal head circumference

https://zenodo.org/records/1322001
https://hc18.grand-challenge.org/

## Dataset structure
Files Structure:
images firectory contains images as a flat list. For training set it also includes annotation images.
Pair of images for train:
```
<image_name>.png
<image_name>_Annotation.png
```
Where <image_name> is just `filename` from CSV without `.png` extension. For test set there will be no `annotation` files.

Sample of directory content for train:
```
872_4HC.png
872_4HC_Annotation.png
875_HC.png
875_HC_Annotation.png
```


CSV Structure
```
filename,pixel size,head circumference (mm)
1_HC.png,0.0691358041432,44.3
2_HC.png,0.0896585200514,56.81
3_HC.png,0.062032562715,68.75
```