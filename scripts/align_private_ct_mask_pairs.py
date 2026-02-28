"""
Build aligned CT-mask pairs for private data.

Scenario:
- CT root layout: <ct_root>/<case_id>/ct.nii.gz
- Mask layout:    <mask_root>/<case_id>.nii.gz

The script checks geometry consistency (size/spacing/origin/direction).
If mismatch exists, it resamples mask to CT geometry (nearest-neighbor),
then writes an aligned mask file and a pairs CSV for downstream training
or validation.

Example:
  python scripts/align_private_ct_mask_pairs.py ^
    --ct_root "D:/ct_data" ^
    --mask_root "D:/nnunet_mask_nii_327" ^
    --out_mask_root "D:/nnunet_mask_nii_327_aligned" ^
    --pairs_csv "D:/nnunet_mask_nii_327_aligned/pairs_aligned.csv"
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def _abs_no_resolve(p: str | Path) -> Path:
    return Path(os.path.abspath(str(Path(p).expanduser())))


def _case_id_from_mask_path(p: Path) -> str:
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if p.suffix:
        return p.stem
    return name


def _same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and a.GetSpacing() == b.GetSpacing()
        and a.GetOrigin() == b.GetOrigin()
        and a.GetDirection() == b.GetDirection()
    )


def _resample_mask_to_image(mask: sitk.Image, image: sitk.Image) -> sitk.Image:
    return sitk.Resample(
        mask,
        image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask.GetPixelID(),
    )


def _to_csv_path_str(p: Path) -> str:
    return str(p).replace("\\", "/")


def main() -> int:
    ap = argparse.ArgumentParser(description="Align private CT-mask pairs and export CSV")
    ap.add_argument("--ct_root", required=True, help="CT root, layout: <case_id>/ct.nii.gz")
    ap.add_argument("--mask_root", required=True, help="Mask root, layout: <case_id>.nii.gz")
    ap.add_argument("--ct_name", default="ct.nii.gz", help="CT filename under each case directory")
    ap.add_argument("--out_mask_root", required=True, help="Directory to write aligned masks")
    ap.add_argument("--pairs_csv", required=True, help="Output pairs CSV path")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing aligned masks")
    args = ap.parse_args()

    ct_root = _abs_no_resolve(args.ct_root)
    mask_root = _abs_no_resolve(args.mask_root)
    out_mask_root = _abs_no_resolve(args.out_mask_root)
    pairs_csv = _abs_no_resolve(args.pairs_csv)

    if not ct_root.is_dir():
        raise SystemExit(f"ct_root not found: {ct_root}")
    if not mask_root.is_dir():
        raise SystemExit(f"mask_root not found: {mask_root}")

    out_mask_root.mkdir(parents=True, exist_ok=True)
    pairs_csv.parent.mkdir(parents=True, exist_ok=True)

    ct_map = {}
    for d in ct_root.iterdir():
        if not d.is_dir():
            continue
        p = d / args.ct_name
        if p.is_file():
            ct_map[d.name] = p

    mask_map = {}
    for p in mask_root.glob("*.nii.gz"):
        mask_map[_case_id_from_mask_path(p)] = p

    common = sorted(set(ct_map) & set(mask_map))
    only_ct = sorted(set(ct_map) - set(mask_map))
    only_mask = sorted(set(mask_map) - set(ct_map))

    n_same = 0
    n_resampled = 0
    n_failed = 0
    n_empty = 0

    with pairs_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "image_path",
                "mask_path",
                "src_mask_path",
                "geometry_status",
                "nonzero_voxels",
            ],
        )
        writer.writeheader()

        for cid in common:
            img_path = ct_map[cid]
            src_mask_path = mask_map[cid]
            out_mask_path = out_mask_root / f"{cid}.nii.gz"

            try:
                img = sitk.ReadImage(str(img_path))
                msk = sitk.ReadImage(str(src_mask_path))

                status = "same_geometry"
                if _same_geometry(img, msk):
                    n_same += 1
                    if args.overwrite or (not out_mask_path.exists()):
                        shutil.copy2(src_mask_path, out_mask_path)
                else:
                    status = "resampled_to_image_geometry"
                    n_resampled += 1
                    aligned = _resample_mask_to_image(msk, img)
                    arr = sitk.GetArrayFromImage(aligned)
                    arr = (arr > 0).astype(np.uint8)
                    aligned_u8 = sitk.GetImageFromArray(arr)
                    aligned_u8.CopyInformation(img)
                    sitk.WriteImage(aligned_u8, str(out_mask_path), useCompression=True)

                out_img = sitk.ReadImage(str(out_mask_path))
                out_arr = sitk.GetArrayFromImage(out_img)
                nonzero = int((out_arr > 0).sum())
                if nonzero == 0:
                    n_empty += 1

                writer.writerow(
                    {
                        "case_id": cid,
                        "image_path": _to_csv_path_str(img_path),
                        "mask_path": _to_csv_path_str(out_mask_path),
                        "src_mask_path": _to_csv_path_str(src_mask_path),
                        "geometry_status": status,
                        "nonzero_voxels": nonzero,
                    }
                )
            except Exception as e:
                n_failed += 1
                writer.writerow(
                    {
                        "case_id": cid,
                        "image_path": _to_csv_path_str(img_path),
                        "mask_path": _to_csv_path_str(out_mask_path),
                        "src_mask_path": _to_csv_path_str(src_mask_path),
                        "geometry_status": f"failed: {e}",
                        "nonzero_voxels": "",
                    }
                )

    print(f"ct_cases: {len(ct_map)}")
    print(f"mask_cases: {len(mask_map)}")
    print(f"paired_cases: {len(common)}")
    print(f"only_ct_no_mask: {len(only_ct)}")
    print(f"only_mask_no_ct: {len(only_mask)}")
    print(f"same_geometry: {n_same}")
    print(f"resampled: {n_resampled}")
    print(f"failed: {n_failed}")
    print(f"empty_masks_after_align: {n_empty}")
    print(f"aligned_mask_root: {out_mask_root}")
    print(f"pairs_csv: {pairs_csv}")

    miss_ct_txt = pairs_csv.parent / "ct_without_mask.txt"
    miss_mask_txt = pairs_csv.parent / "mask_without_ct.txt"
    miss_ct_txt.write_text("\n".join(only_ct), encoding="utf-8")
    miss_mask_txt.write_text("\n".join(only_mask), encoding="utf-8")
    print(f"saved: {miss_ct_txt}")
    print(f"saved: {miss_mask_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
