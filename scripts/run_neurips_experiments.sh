#!/bin/bash
set -euo pipefail

cd /autodl-tmp/mamba-sci
mkdir -p logs /autodl-tmp/mamba-sci/checkpoints/neurips_vim_3d /autodl-tmp/mamba-sci/checkpoints/neurips_transformer_3d

# Force 3D token layout to 164 (global=64 + local=100) via paths.yaml.
# train_vlm.py reads these from config/paths.yaml.
/root/miniconda3/envs/mamba_a800/bin/python - <<'PY'
from pathlib import Path
import yaml
p = Path('/autodl-tmp/mamba-sci/config/paths.yaml')
d = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
d['global_pool_size_3d'] = [4, 4, 4]   # 64
d['local_crop_size_3d'] = [2, 5, 10]    # 100
d['patch_size_3d'] = [32, 128, 128]
p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding='utf-8')
print('Updated paths.yaml 3D token config to 164 total.')
PY

# Safety: clear stale sessions with same names
screen -S mamba_run -X quit >/dev/null 2>&1 || true
screen -S transformer_run -X quit >/dev/null 2>&1 || true

# Experiment 1: Mamba/Vim Bridge (Proposed Method)
screen -dmS mamba_run bash -lc '
  cd /autodl-tmp/mamba-sci
  /root/miniconda3/envs/mamba_a800/bin/python -u train_vlm.py \
    --spatial_dims 3 \
    --patch_size_3d 32,128,128 \
    --vision_bridge_type vim \
    --mamba_model /autodl-tmp/models/mamba-2.8b-hf \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_workers 0 \
    --bf16 True \
    --max_visual_tokens 164 \
    --save_every_steps 0 \
    --output_dir /autodl-tmp/mamba-sci/checkpoints/neurips_vim_3d \
    --epochs 50 \
    --lr 1e-4 \
  > logs/train_vim_3d.log 2>&1
'

# Experiment 2: Transformer Bridge (Baseline)
# Queue mode: start only after Vim run exits (avoids same-GPU contention/OOM overnight)
screen -dmS transformer_run bash -lc '
  cd /autodl-tmp/mamba-sci
  while pgrep -af "train_vlm.py" | grep -q "neurips_vim_3d"; do
    sleep 60
  done
  /root/miniconda3/envs/mamba_a800/bin/python -u train_vlm.py \
    --spatial_dims 3 \
    --patch_size_3d 32,128,128 \
    --vision_bridge_type transformer \
    --mamba_model /autodl-tmp/models/mamba-2.8b-hf \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_workers 0 \
    --bf16 True \
    --max_visual_tokens 164 \
    --save_every_steps 0 \
    --output_dir /autodl-tmp/mamba-sci/checkpoints/neurips_transformer_3d \
    --epochs 50 \
    --lr 1e-4 \
  > logs/train_transformer_3d.log 2>&1
'

echo "Experiments launched in detached screens: mamba_run, transformer_run"
echo "Transformer run is queued and will start after Vim run exits."
echo "Check logs at logs/train_vim_3d.log and logs/train_transformer_3d.log"
screen -ls || true
