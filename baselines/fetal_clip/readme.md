## Usage

Build:
```
docker build -t fetal-clip baselines/fetal_clip/
```

Run:
```
docker build -t fetal-clip baselines/fetal_clip/
```

```
docker run \
  -e DIR_IMAGES=/mnt/images \
  -e PATH_CSV=/mnt/data.csv \
  -e BATCH_SIZE=32 \
  -v /Users/pmromanchenko/Downloads/FETAL_PLANES_ZENODO/Images:/mnt/images \
  -v /Users/pmromanchenko/private/ultrasound-analysis/baselines/fetal_clip/fetal_planes.csv:/mnt/data.csv \
  fetal-clip
```