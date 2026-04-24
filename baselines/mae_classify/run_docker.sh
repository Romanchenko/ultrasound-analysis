 docker run --rm --gpus all \
     -v /home/p.romanchenko/elle_data/fomina_rgb:/data/images:ro \
     -v /home/p.romanchenko/experiments/mae-classify/processed_iter_1.csv:/data/processed_iter_1.csv:ro \
     -v /home/p.romanchenko/experiments/mae-classify/weights:/weights:ro \
     -v /home/p.romanchenko/experiments/mae-classify/results/v12:/app/results \
     -e EPOCHS=60 \
     -e LR=1e-4 \
     -e PATH_MAE_CHECKPOINT=/weights/mae_v12.pt \
     -e HEAD_HIDDEN_DIM=512 \
     -e EXCLUDE_CLASS_IDS="" \
     -e BATCH_SIZE=32 \
     -e NUM_WORKERS=2 \
     -e CONFIG_PATH=/app/results/run_config.json \
     -e DEVICE=cuda:1 \
     promanchenko-mae-classify:latest

docker run --rm \
     -v /home/p.romanchenko/elle_data/fomina_rgb:/data/images:ro \
     -v /home/p.romanchenko/experiments/mae-classify/processed_iter_1.csv:/data/processed_iter_1.csv:ro \
     -v /home/p.romanchenko/experiments/mae-classify/weights:/weights:ro \
     -v /home/p.romanchenko/experiments/mae-classify/results/v10:/app/results \
     -e EPOCHS=30 \
     -e LR=1e-4 \
     -e PATH_MAE_CHECKPOINT=/weights/mae_v10.pt \
     -e HEAD_HIDDEN_DIM=512 \
     -e EXCLUDE_CLASS_IDS="" \
     -e BATCH_SIZE=32 \
     -e NUM_WORKERS=2 \
     -e EVAL_ONLY=1 \
     -e PATH_HEAD_CHECKPOINT=/app/results/best_head.pt \
     -e CONFIG_PATH=/app/results/run_config.json \
     promanchenko-mae-classify:latest