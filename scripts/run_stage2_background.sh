#!/bin/bash
# Stage2 后台训练：在 screen 内 cd 到项目目录并运行，避免 cwd/路径问题
# 用法: bash scripts/run_stage2_background.sh
# 或直接复制下方 screen 整段命令执行

set -e
OUT_DIR="${OUT_DIR:-/autodl-tmp/outputs/stage2_private_v5_ord}"
mkdir -p "$OUT_DIR"
cd /autodl-tmp/mamba-sci
exec /root/miniconda3/envs/mamba_a800/bin/python train_vlm.py \
  --csv /autodl-tmp/caption_train_private_v4_grade4_auto.csv \
  --epochs 30 \
  --batch_size 4 \
  --num_workers 8 \
  --lr 1e-5 \
  --max_visual_tokens 144 \
  --max_text_len 640 \
  --save_every_steps 100 \
  --log_every_steps 20 \
  --mamba_model /autodl-tmp/models/mamba-2.8b-hf \
  --output_dir "$OUT_DIR" \
  --lambda_cls 1.0
