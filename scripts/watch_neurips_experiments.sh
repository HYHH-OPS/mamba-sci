#!/bin/bash
set -euo pipefail
cd /autodl-tmp/mamba-sci

echo "=== Screen sessions ==="
screen -ls || true

echo

echo "=== Running train processes ==="
pgrep -af "train_vlm.py" || echo "No train_vlm.py process running"

echo

echo "=== GPU (nvidia-smi) ==="
nvidia-smi || true

echo

echo "=== vim log (tail 60) ==="
tail -n 60 logs/train_vim_3d.log 2>/dev/null || echo "logs/train_vim_3d.log not found yet"

echo

echo "=== transformer log (tail 60) ==="
tail -n 60 logs/train_transformer_3d.log 2>/dev/null || echo "logs/train_transformer_3d.log not found yet"
