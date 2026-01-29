# FETAL_PLANES_DB: Common maternal-fetal ultrasound images

https://zenodo.org/records/3904280
https://www.nature.com/articles/s41598-020-67076-5


## Dataset file structure
```
FETAL_PLANES_DB_data.csv
FETAL_PLANES_DB_data.xlsx
images/
    <image_name>.png
```

CSV structure:
```
Image_name;                     Patient_num;    Plane;  Brain_plane;    Operator;   US_Machine; Train 
Patient00001_Plane1_1_of_15;    1;              Other;  Not A Brain;    Other;      Aloka;      1 
Patient00001_Plane1_2_of_15;    1;              Other;  Not A Brain;    Other;      Aloka;      1 
Patient00001_Plane1_3_of_15;    1;              Other;  Not A Brain;    Other;      Aloka;      1 
```
Total number of images: 12400

## Validation model
Для валидации нацелимся на задачу классификации Brain_plane

## Basic tranforms
- привести к одинаковому размеру 
- убрать текст с картинки (на каждом снимке в правом верхнем углу есть метаинформация, зависящая от конкретного прибора)