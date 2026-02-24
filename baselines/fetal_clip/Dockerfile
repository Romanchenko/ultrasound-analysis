# ---------- build-time: download weights & install deps ----------
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir gdown

# Download FetalCLIP weights into /weights/
RUN mkdir -p /weights && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -fSL -o /weights/FetalCLIP_weights.pt \
        "https://huggingface.co/numansaeed/fetalclip-model/resolve/main/FetalCLIP_weights.pt" && \
    apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# ---------- runtime image ----------
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model weights from builder
COPY --from=builder /weights/FetalCLIP_weights.pt /weights/FetalCLIP_weights.pt

# Copy application code & bundled config files
COPY run.py .
COPY fetal_clip_config.json .
COPY classes.json .

# ----- runtime-configurable paths (override with `docker run -e`) -----
# Where images live (must be mounted into the container)
ENV DIR_IMAGES="/data/images"
# CSV with class/image/class_idx columns (must be mounted)
ENV PATH_CSV="/data/processed_iter_1.csv"
# Downloaded at build time; override only if you mount your own weights
ENV PATH_FETALCLIP_WEIGHT="/weights/FetalCLIP_weights.pt"
# Bundled inside the image; override if you mount a custom config
ENV PATH_FETALCLIP_CONFIG="/app/fetal_clip_config.json"
# Bundled inside the image; override if you mount custom prompts
ENV PATH_TEXT_PROMPTS="/app/classes.json"
# Other tunables
ENV BATCH_SIZE="16"
ENV NUM_WORKERS="4"
ENV IMAGE_EXT=""

ENTRYPOINT ["python", "run.py"]

