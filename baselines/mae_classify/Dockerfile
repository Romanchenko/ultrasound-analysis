# MAE / CNN-ViT classification baseline
#
# Build context must be the repository root so that embeddings/ is reachable:
#
#   docker build -t mae-classify -f baselines/mae_classify/Dockerfile .
#
# Run plain MAE-ViT classifier:
#
#   docker run --rm --gpus all \
#     -v /host/path/to/npy:/data/images:ro \
#     -v /host/path/to/processed_iter_1.csv:/data/processed_iter_1.csv:ro \
#     -v /host/path/to/checkpoints:/weights:ro \
#     -v /host/path/to/output:/app/results \
#     -e EPOCHS=50 -e LR=1e-3 \
#     mae-classify
#
# Run CNN-ViT classifier (override entrypoint):
#
#   docker run --rm --gpus all ... \
#     --entrypoint python mae-classify /app/run_cnn_vit.py

FROM python:3.11-slim

WORKDIR /app

# Python dependencies
COPY baselines/mae_classify/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the plain MAE-ViT encoder package
RUN mkdir -p embeddings/vit
RUN touch embeddings/__init__.py
COPY embeddings/vit/__init__.py      embeddings/vit/__init__.py
COPY embeddings/vit/model.py         embeddings/vit/model.py
COPY embeddings/vit/train.py         embeddings/vit/train.py

# Copy the CNN-ViT encoder package
RUN mkdir -p embeddings/cnn_vit
COPY embeddings/cnn_vit/__init__.py  embeddings/cnn_vit/__init__.py
COPY embeddings/cnn_vit/model.py     embeddings/cnn_vit/model.py
COPY embeddings/cnn_vit/train.py     embeddings/cnn_vit/train.py

# Copy the image_clenup preprocessing package
RUN mkdir -p image_clenup/conus_detection
RUN touch image_clenup/__init__.py
COPY image_clenup/__init__.py                        image_clenup/__init__.py
COPY image_clenup/pipeline.py                        image_clenup/pipeline.py
COPY image_clenup/conus_detection/__init__.py        image_clenup/conus_detection/__init__.py
COPY image_clenup/conus_detection/model.py           image_clenup/conus_detection/model.py
COPY image_clenup/conus_detection/train.py           image_clenup/conus_detection/train.py

# Copy both classification scripts
COPY baselines/mae_classify/run.py          run.py
COPY baselines/mae_classify/run_cnn_vit.py  run_cnn_vit.py

# ----- mount points -----
#   /data/images              – .npy image files  (read-only)
#   /data/processed_iter_1.csv – labels CSV        (read-only)
#   /weights                  – MAE checkpoint(s)  (read-only)
#   /app/results              – output artifacts   (read-write)
VOLUME ["/data/images", "/data/processed_iter_1.csv", "/weights", "/app/results"]

# ----- runtime-configurable env vars (override with `docker run -e`) -----
ENV DIR_IMAGES="/data/images"
ENV PATH_CSV="/data/processed_iter_1.csv"
ENV PATH_MAE_CHECKPOINT="/weights/mae_final.pt"
ENV IMAGE_EXT=""
ENV EXCLUDE_CLASS_IDS="4"
ENV EPOCHS="50"
ENV BATCH_SIZE="64"
ENV LR="1e-3"
ENV WEIGHT_DECAY="1e-4"
ENV VAL_SPLIT="0.2"
ENV HEAD_HIDDEN_DIM="0"
ENV HEAD_DROPOUT="0.1"
ENV NUM_WORKERS="4"
ENV MAX_IMAGE_HEIGHT="224"
# Per-image z-score standardisation; must match MAE pre-training transform.
# Set to "0" only when evaluating a legacy MAE checkpoint trained without it.
ENV STANDARDIZE_INPUT="1"
ENV SEED="42"
ENV EVAL_ONLY="0"
ENV PATH_HEAD_CHECKPOINT="best_head.pt"
ENV CONFIG_PATH="run_config.json"
# Per-epoch metrics CSV (train/val loss, acc, balanced acc).
ENV METRICS_CSV="training_metrics.csv"
ENV MODE="train"
ENV DEVICE=""
ENV RANDOM_ENCODER="0"
ENV PREPROCESS_IMAGES="0"
ENV CONUS_CHECKPOINT=""

WORKDIR /app/results

ENTRYPOINT ["python", "/app/run.py"]
