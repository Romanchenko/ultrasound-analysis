 docker run --rm --gpus device=GPU-c9c85b90-54f9-5ead-e9c1-9db1f1f30a4d \
     -v /home/p.romanchenko/elle_data/fomina_rgb:/data/images:ro \
     -v /home/p.romanchenko/experiments/mae-classify/processed_iter_1.csv:/data/processed_iter_1.csv:ro \
     -v /home/p.romanchenko/experiments/mae-classify/weights:/weights:ro \
     -v /home/p.romanchenko/experiments/mae-classify/results/v11:/app/results \
     -e EPOCHS=10 \
     -e LR=1e-4 \
     -e PATH_MAE_CHECKPOINT=/weights/mae_v11.pt \
     -e HEAD_HIDDEN_DIM=512 \
     -e EXCLUDE_CLASS_IDS="" \
     apollin/mae-classify:v1.2