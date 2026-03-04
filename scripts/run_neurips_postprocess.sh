#!/bin/bash
set -euo pipefail

REPO=/autodl-tmp/mamba-sci
PY=/root/miniconda3/envs/mamba_a800/bin/python
MAMBA_MODEL=/autodl-tmp/models/mamba-2.8b-hf

VIM_OUT="$REPO/checkpoints/neurips_vim_3d"
TR_OUT="$REPO/checkpoints/neurips_transformer_3d"

ORACLE_CSV=/autodl-tmp/caption_val_private_oracle.csv
DETECTED_CSV=/autodl-tmp/caption_val_private_detected.csv
BASE_VAL=/autodl-tmp/caption_val_private_v5_ord_all_clean.csv

OUT_ROOT=/autodl-tmp/mamba-res
OUT_WORK="$OUT_ROOT/neurips_postprocess"
OUT_ORACLE="$OUT_ROOT/neurips_oracle_eval"
OUT_DETECTED="$OUT_ROOT/neurips_detected_eval"
OUT_TR_ORACLE="$OUT_ROOT/neurips_transformer_eval_oracle"

mkdir -p "$OUT_WORK" "$OUT_ORACLE" "$OUT_DETECTED" "$OUT_TR_ORACLE"

echo "[post] waiting for Vim training process to finish..."
while pgrep -af "train_vlm.py" | grep -q "neurips_vim_3d"; do
  sleep 120
done

echo "[post] waiting for Transformer training process to finish..."
while pgrep -af "train_vlm.py" | grep -q "neurips_transformer_3d"; do
  sleep 120
done

VIM_CKPT="$VIM_OUT/vision_bridge_vlm_final.pt"
TR_CKPT="$TR_OUT/vision_bridge_vlm_final.pt"

if [ ! -f "$VIM_CKPT" ]; then
  echo "[post][error] Vim checkpoint missing: $VIM_CKPT"
  exit 1
fi

if [ ! -f "$TR_CKPT" ]; then
  echo "[post][warn] Transformer checkpoint missing: $TR_CKPT (will skip transformer eval)"
fi

echo "[post] building Oracle/Detected CSV..."
"$PY" - <<'PY'
import os
import pandas as pd

base = "/autodl-tmp/caption_val_private_v5_ord_all_clean.csv"
oracle_csv = "/autodl-tmp/caption_val_private_oracle.csv"
detected_csv = "/autodl-tmp/caption_val_private_detected.csv"

df = pd.read_csv(base)
oracle = df.copy()
detected = df.copy()

src = "/root/autodl-tmp/datasets/private_masks_aligned"
dst = "/root/autodl-tmp/datasets/private_masks_raw"
detected["mask_path"] = detected["mask_path"].astype(str).str.replace(src, dst, regex=False)

oracle.to_csv(oracle_csv, index=False, encoding="utf-8-sig")
detected.to_csv(detected_csv, index=False, encoding="utf-8-sig")

for tag, path in [("oracle", oracle_csv), ("detected", detected_csv)]:
    t = pd.read_csv(path)
    miss = (~t["mask_path"].astype(str).map(os.path.exists)).sum()
    print(f"{tag}: rows={len(t)}, missing_masks={int(miss)}")
PY

PATHS_YAML="$REPO/config/paths.yaml"
PATHS_BAK="$OUT_WORK/paths.yaml.before_oracle_detected"
cp "$PATHS_YAML" "$PATHS_BAK"

set_val_csv() {
  local csv_path="$1"
  "$PY" - <<PY
from pathlib import Path
import yaml
p = Path("$PATHS_YAML")
d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
d["caption_csv_val"] = "$csv_path"
p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")
print("caption_csv_val ->", d["caption_csv_val"])
PY
}

run_bundle() {
  local ckpt="$1"
  local out_dir="$2"
  local train_log="$3"
  local num_val="$4"
  "$PY" "$REPO/scripts/run_private_eval_bundle.py" \
    --checkpoint "$ckpt" \
    --mamba_model "$MAMBA_MODEL" \
    --out_dir "$out_dir" \
    --num_val "$num_val" \
    --max_visual_tokens 164 \
    --max_new_tokens 512 \
    --num_beams 4 \
    --length_penalty 1.1 \
    --repetition_penalty 1.0 \
    --no_repeat_ngram_size 0 \
    --suppress_eos_steps 128 \
    --train_log "$train_log"
}

echo "[post] running Vim Oracle eval..."
set_val_csv "$ORACLE_CSV"
run_bundle "$VIM_CKPT" "$OUT_ORACLE" "$VIM_OUT/stage2_train_log.csv" 23

echo "[post] running Vim Detected eval..."
set_val_csv "$DETECTED_CSV"
run_bundle "$VIM_CKPT" "$OUT_DETECTED" "$VIM_OUT/stage2_train_log.csv" 23

if [ -f "$TR_CKPT" ]; then
  echo "[post] running Transformer Oracle eval..."
  set_val_csv "$ORACLE_CSV"
  run_bundle "$TR_CKPT" "$OUT_TR_ORACLE" "$TR_OUT/stage2_train_log.csv" 23
fi

cp "$PATHS_BAK" "$PATHS_YAML"

echo "[post] summarizing metrics..."
"$PY" - <<'PY'
import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path("/autodl-tmp/mamba-res")
WORK = ROOT / "neurips_postprocess"

def latest_run_analysis(base: Path):
    runs = sorted([p for p in base.glob("run_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not runs:
        return None, None
    r = runs[-1]
    s = r / "analysis" / "summary.json"
    if not s.exists():
        return r, None
    return r, json.loads(s.read_text(encoding="utf-8"))

def parse_log(log_path: Path):
    txt = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    sps = [float(x) for x in re.findall(r"sps=([0-9]+(?:\.[0-9]+)?)", txt)]
    mem = [float(x) for x in re.findall(r"GPU.*?([0-9]+(?:\.[0-9]+)?) GB", txt)]
    return {
        "log_path": str(log_path),
        "sps_mean": mean(sps[-200:]) if sps else None,
        "sps_last": sps[-1] if sps else None,
        "mem_gb_first": mem[0] if mem else None,
        "mem_gb_last": mem[-1] if mem else None,
    }

vim_oracle_run, vim_oracle = latest_run_analysis(ROOT / "neurips_oracle_eval")
vim_detected_run, vim_detected = latest_run_analysis(ROOT / "neurips_detected_eval")
tr_oracle_run, tr_oracle = latest_run_analysis(ROOT / "neurips_transformer_eval_oracle")

out = {
    "vim_oracle_run": str(vim_oracle_run) if vim_oracle_run else None,
    "vim_detected_run": str(vim_detected_run) if vim_detected_run else None,
    "transformer_oracle_run": str(tr_oracle_run) if tr_oracle_run else None,
    "vim_oracle_summary": vim_oracle,
    "vim_detected_summary": vim_detected,
    "transformer_oracle_summary": tr_oracle,
    "vim_train_profile": parse_log(Path("/autodl-tmp/mamba-sci/logs/train_vim_3d.log")),
    "transformer_train_profile": parse_log(Path("/autodl-tmp/mamba-sci/logs/train_transformer_3d.log")),
}

if vim_oracle and vim_detected:
    gap = {}
    for k in ["section_all_ok_rate", "hallucination_rate", "classification_accuracy", "contour_success_rate"]:
        a = vim_oracle.get(k)
        b = vim_detected.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            gap[k + "_delta_detected_minus_oracle"] = float(b - a)
    out["oracle_vs_detected_gap"] = gap

WORK.mkdir(parents=True, exist_ok=True)
(WORK / "oracle_detected_gap.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

lines = []
lines.append("# NeurIPS Vim vs Transformer + Oracle vs Detected Summary")
lines.append("")
def one(name, s):
    if not s:
        lines.append(f"- {name}: N/A")
        return
    lines.append(f"- {name}: section_all_ok={s.get('section_all_ok_rate')}, hallucination={s.get('hallucination_rate')}, cls_acc={s.get('classification_accuracy')}, contour_ok={s.get('contour_success_rate')}")
one("Vim Oracle", vim_oracle)
one("Vim Detected", vim_detected)
one("Transformer Oracle", tr_oracle)
if "oracle_vs_detected_gap" in out:
    lines.append("")
    lines.append("## Oracle vs Detected Gap (Detected - Oracle)")
    for k, v in out["oracle_vs_detected_gap"].items():
        lines.append(f"- {k}: {v:+.6f}")
lines.append("")
lines.append("## Throughput")
for tag, p in [("Vim", out["vim_train_profile"]), ("Transformer", out["transformer_train_profile"])]:
    lines.append(f"- {tag}: sps_mean={p.get('sps_mean')}, sps_last={p.get('sps_last')}, mem_first={p.get('mem_gb_first')} GB")
(WORK / "oracle_detected_gap.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote:", WORK / "oracle_detected_gap.json")
print("wrote:", WORK / "oracle_detected_gap.md")
PY

STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_DIR="$OUT_ROOT/neurips_paper_assets_$STAMP"
mkdir -p "$BUNDLE_DIR"

cp -f "$REPO/scripts/run_neurips_experiments.sh" "$BUNDLE_DIR/" || true
cp -f "$REPO/scripts/run_neurips_postprocess.sh" "$BUNDLE_DIR/" || true
cp -f "$REPO/logs/train_vim_3d.log" "$BUNDLE_DIR/" || true
cp -f "$REPO/logs/train_transformer_3d.log" "$BUNDLE_DIR/" || true
cp -f "$VIM_OUT/stage2_train_log.csv" "$BUNDLE_DIR/" || true
cp -f "$VIM_OUT/stage2_epoch_metrics.csv" "$BUNDLE_DIR/" || true
cp -f "$TR_OUT/stage2_train_log.csv" "$BUNDLE_DIR/" || true
cp -f "$TR_OUT/stage2_epoch_metrics.csv" "$BUNDLE_DIR/" || true
cp -f "$ORACLE_CSV" "$BUNDLE_DIR/" || true
cp -f "$DETECTED_CSV" "$BUNDLE_DIR/" || true
cp -f "$OUT_WORK/oracle_detected_gap.json" "$BUNDLE_DIR/" || true
cp -f "$OUT_WORK/oracle_detected_gap.md" "$BUNDLE_DIR/" || true
cp -f "$PATHS_BAK" "$BUNDLE_DIR/" || true

for d in "$OUT_ORACLE" "$OUT_DETECTED" "$OUT_TR_ORACLE"; do
  if [ -d "$d" ]; then
    latest="$(ls -dt "$d"/run_* 2>/dev/null | head -n 1 || true)"
    if [ -n "${latest:-}" ]; then
      cp -r "$latest" "$BUNDLE_DIR/"
    fi
  fi
done

TAR="$OUT_ROOT/neurips_paper_assets_$STAMP.tar.gz"
tar -czf "$TAR" -C "$OUT_ROOT" "$(basename "$BUNDLE_DIR")"

echo "[post] bundle ready: $TAR"
echo "$TAR" > "$OUT_WORK/latest_bundle_path.txt"

