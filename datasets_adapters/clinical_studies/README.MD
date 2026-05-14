Dataset Clinical Ultrasound Image Repository


https://clinical-ultrasound-image-repository.s3.amazonaws.com/index.html

File structure:

```
/root_dir
    /meta
        all-meta.json
    /studies
        /{STUDY_NAME}.zip
        ...
```
Here STUDY_NAME is just a string describing type of analysys, maybe patient number too -- this is not that important for us right now.

Each zip archive contains several DICOM files:
```
/A10026_US_RETROPERITONEAL_COMPLETE__2901241556535691
    /meta
        /json
            /A10026-meta.json
        /csv
            /A10026-meta.csv
    /images
        /dicom
            /A10026-1-8-5epdlg.dcm
            /A10026-1-7-hugz34.dcm
            ...
```

Content of A10026-meta.json for instance:
```json
{
  "A10026": {
    "_ref": "A10026_US_RETROPERITONEAL_COMPLETE__2901241556535691", 
    "demo": {
      "BirthYear": "1935", 
      "DeathYear": " ", 
      "Ethnicity": "Non-Hispanic", 
      "FirstRace": "White", 
      "MultiRacial": "No", 
      "Sex": "Male"
    }, 
    "exam": {
      "AgeAtExam": "84", 
      "IsAbnormal": "No", 
      "ProcedureName": "US RETROPERITONEAL COMPLETE (76770)", 
      "SmokingStatus": "Never Smoker", 
      "Status": "Alive"
    }, 
    "labs": {
      "BloodType": "B Positive"
    }
  }
}
```


Preprocessed files structure:
```
/root_dir
    /meta
        all-meta.json
    /studies
        /{STUDY_NAME}.zip
        ...
    /studies_first_frame
        /{STUDY_NAME}
          /{STUDY_NAME}
            /meta
              /json
                  /{STUDY_ID}-meta.json
              /csv
                  /{STUDY_ID}-meta.csv
            /images
              /dicom
                /{IMAGE_NAME}.jpg
                ...
```