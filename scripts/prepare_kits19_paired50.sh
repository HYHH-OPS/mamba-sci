#!/usr/bin/env bash
set -euo pipefail

# Build a paired kits19 subset by linking:
# - imaging from /root/autodl-tmp/datasets/kits19/data/case_xxxxx
# - segmentation from /root/datasets/kits19/data/case_xxxxx
#
# Output:
#   /autodl-tmp/datasets/kits19_paired50/case_xxxxx/{imaging.nii.gz,segmentation.nii.gz}
#   /autodl-tmp/datasets/kits19_paired50/paired_cases.csv

SRC_IMG_ROOT="${1:-/root/autodl-tmp/datasets/kits19/data}"
SRC_SEG_ROOT="${2:-/root/datasets/kits19/data}"
OUT_ROOT="${3:-/autodl-tmp/datasets/kits19_paired50}"

mkdir -p "${OUT_ROOT}"
CSV="${OUT_ROOT}/paired_cases.csv"
echo "case,image,mask" > "${CSV}"

paired=0
missing=0

shopt -s nullglob
for d in "${SRC_IMG_ROOT}"/case_*; do
  c="$(basename "${d}")"
  img=""
  seg=""

  if [[ -f "${d}/imaging.nii.gz" ]]; then
    img="${d}/imaging.nii.gz"
  elif [[ -f "${d}/imaging.nii" ]]; then
    img="${d}/imaging.nii"
  fi

  if [[ -f "${SRC_SEG_ROOT}/${c}/segmentation.nii.gz" ]]; then
    seg="${SRC_SEG_ROOT}/${c}/segmentation.nii.gz"
  elif [[ -f "${SRC_SEG_ROOT}/${c}/segmentation.nii" ]]; then
    seg="${SRC_SEG_ROOT}/${c}/segmentation.nii"
  fi

  if [[ -n "${img}" && -n "${seg}" ]]; then
    out_case="${OUT_ROOT}/${c}"
    mkdir -p "${out_case}"
    ln -sfn "${img}" "${out_case}/imaging.nii.gz"
    ln -sfn "${seg}" "${out_case}/segmentation.nii.gz"
    echo "${c},${img},${seg}" >> "${CSV}"
    paired=$((paired + 1))
  else
    missing=$((missing + 1))
  fi
done

echo "paired=${paired} missing=${missing}"
echo "csv=${CSV}"
