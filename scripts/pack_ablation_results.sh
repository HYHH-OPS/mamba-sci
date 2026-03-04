#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

ASSET_DIR="ablation_paper_assets"
OUT_TAR="ablation_results_final.tar.gz"

rm -rf "${ASSET_DIR}"
mkdir -p "${ASSET_DIR}/logs" "${ASSET_DIR}/manifests" "${ASSET_DIR}/plots"

# 1) Metrics and tables
find logs -maxdepth 1 -type f \( -name "*.json" -o -name "*.csv" \) -print0 2>/dev/null | while IFS= read -r -d '' f; do
  cp -f "$f" "${ASSET_DIR}/logs/"
done

# 2) Data manifests only (no image volumes)
for f in \
  /autodl-tmp/public_data/rex_val/rex_val.csv \
  /autodl-tmp/public_data/lidc_val/lidc_val.csv
do
  if [[ -f "$f" ]]; then
    cp -f "$f" "${ASSET_DIR}/manifests/"
  fi
done

# 3) Any generated png figures
find logs -maxdepth 2 -type f -name "*.png" -print0 2>/dev/null | while IFS= read -r -d '' f; do
  cp -f "$f" "${ASSET_DIR}/plots/"
done
find /autodl-tmp/mamba-res/public_ablation -type f -name "*.png" -print0 2>/dev/null | while IFS= read -r -d '' f; do
  cp -f "$f" "${ASSET_DIR}/plots/"
done

tar -czf "${OUT_TAR}" "${ASSET_DIR}"

echo "[done] packed: ${REPO_DIR}/${OUT_TAR}"
echo
echo "Run this on your local machine to download:"
echo "scp -P 53947 root@connect.nma1.seetacloud.com:${REPO_DIR}/${OUT_TAR} D:/mamba/downloads/"

