#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

mkdir -p logs
OUT_ROOT="${OUT_ROOT:-/autodl-tmp/mamba-res/public_ablation}"
mkdir -p "${OUT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mamba_a800/bin/python}"
CHECKPOINT="${CHECKPOINT:-/autodl-tmp/mamba-sci/outputs/vision_bridge_vlm_final.pt}"
MAMBA_MODEL="${MAMBA_MODEL:-/autodl-tmp/models/mamba-2.8b-hf}"
REX_CSV="${REX_CSV:-/autodl-tmp/public_data/rex_val/rex_val.csv}"
LIDC_CSV="${LIDC_CSV:-/autodl-tmp/public_data/lidc_val/lidc_val.csv}"
NUM_VAL="${NUM_VAL:-200}"
JITTER_VOX="${JITTER_VOX:-6}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[error] checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

csv_has_rows() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    return 1
  fi
  local n
  n="$(wc -l < "$p" || echo 0)"
  [[ "${n}" -gt 1 ]]
}

latest_run_dir() {
  local base="$1"
  "${PYTHON_BIN}" - "$base" <<'PY'
import sys
from pathlib import Path
base = Path(sys.argv[1])
runs = sorted([p for p in base.glob("run_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
if not runs:
    raise SystemExit(2)
print(str(runs[-1]))
PY
}

analyze_run_dir() {
  local run_dir="$1"
  local out_dir="$2"
  "${PYTHON_BIN}" scripts/run_private_eval_bundle.py \
    --skip_infer \
    --run_dir "${run_dir}" \
    --out_dir "${out_dir}" \
    >/dev/null
}

run_infer_case() {
  local tag="$1"
  local csv_path="$2"
  local mode="$3"
  local jitter="${4:-0}"
  local out_dir="${OUT_ROOT}/${tag}"
  mkdir -p "${out_dir}"

  echo "[run] ${tag} | mode=${mode} | jitter=${jitter} | csv=${csv_path}"
  "${PYTHON_BIN}" inference.py \
    --val_sample \
    --num_val "${NUM_VAL}" \
    --csv "${csv_path}" \
    --ablation_mode "${mode}" \
    --checkpoint "${CHECKPOINT}" \
    --mamba_model "${MAMBA_MODEL}" \
    --max_visual_tokens 164 \
    --out_dir "${out_dir}" \
    --no_do_sample \
    --roi_jitter_3d "${jitter}"

  local run_dir
  run_dir="$(latest_run_dir "${out_dir}")"
  analyze_run_dir "${run_dir}" "${out_dir}"
  echo "${run_dir}"
}

# Experiment A: GL-FPB validity on ReXGroundingCT
if ! csv_has_rows "${REX_CSV}"; then
  echo "[warn] REX csv not found, skip Experiment A: ${REX_CSV}"
else
  rex_global="$(run_infer_case "rex_global_only" "${REX_CSV}" "global_only" 0)"
  rex_local="$(run_infer_case "rex_local_only" "${REX_CSV}" "local_only" 0)"
  rex_full="$(run_infer_case "rex_full" "${REX_CSV}" "full" 0)"

  "${PYTHON_BIN}" - "${rex_global}" "${rex_local}" "${rex_full}" <<'PY'
import json, sys
from pathlib import Path

def load_summary(run_dir: str):
    p = Path(run_dir) / "analysis" / "summary.json"
    if not p.exists():
        return {}
    s = json.loads(p.read_text(encoding="utf-8"))
    keep = [
        "num_samples",
        "section_all_ok_rate",
        "hallucination_rate",
        "classification_accuracy",
        "classification_valid_n",
        "gen_len_mean",
        "ref_len_mean",
    ]
    return {k: s.get(k) for k in keep}

out = {
    "global_only": {"run_dir": sys.argv[1], "metrics": load_summary(sys.argv[1])},
    "local_only": {"run_dir": sys.argv[2], "metrics": load_summary(sys.argv[2])},
    "full": {"run_dir": sys.argv[3], "metrics": load_summary(sys.argv[3])},
}
log_path = Path("logs/rex_ablation_results.json")
log_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[done] wrote {log_path}")
PY
fi

# Experiment B: ROI robustness on LIDC-IDRI
if ! csv_has_rows "${LIDC_CSV}"; then
  echo "[warn] LIDC csv not found, skip Experiment B: ${LIDC_CSV}"
else
  lidc_oracle="$(run_infer_case "lidc_oracle" "${LIDC_CSV}" "full" 0)"
  lidc_jitter="$(run_infer_case "lidc_detected_jitter" "${LIDC_CSV}" "full" "${JITTER_VOX}")"

  "${PYTHON_BIN}" - "${lidc_oracle}" "${lidc_jitter}" <<'PY'
import json, sys
from pathlib import Path

def load_summary(run_dir: str):
    p = Path(run_dir) / "analysis" / "summary.json"
    if not p.exists():
        return {}
    s = json.loads(p.read_text(encoding="utf-8"))
    keep = [
        "num_samples",
        "section_all_ok_rate",
        "hallucination_rate",
        "classification_accuracy",
        "classification_valid_n",
        "gen_len_mean",
        "ref_len_mean",
    ]
    return {k: s.get(k) for k in keep}

oracle = load_summary(sys.argv[1])
jitter = load_summary(sys.argv[2])

def maybe_gap(k):
    a, b = oracle.get(k), jitter.get(k)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return b - a
    return None

out = {
    "oracle": {"run_dir": sys.argv[1], "metrics": oracle},
    "detected_jitter": {"run_dir": sys.argv[2], "metrics": jitter},
    "gap_detected_minus_oracle": {
        "section_all_ok_rate": maybe_gap("section_all_ok_rate"),
        "hallucination_rate": maybe_gap("hallucination_rate"),
        "classification_accuracy": maybe_gap("classification_accuracy"),
    },
}
log_path = Path("logs/lidc_robustness_results.json")
log_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[done] wrote {log_path}")
PY
fi

echo "[done] public ablation run complete."
echo " - logs/rex_ablation_results.json"
echo " - logs/lidc_robustness_results.json"
