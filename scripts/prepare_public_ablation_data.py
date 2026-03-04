#!/usr/bin/env python3
"""
Prepare mini public ablation datasets on storage-limited servers.

Outputs:
  /autodl-tmp/public_data/rex_val/rex_val.csv
  /autodl-tmp/public_data/lidc_val/lidc_val.csv

The script is intentionally streaming-friendly:
  download -> process -> save lightweight patch -> delete raw file
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _ensure_pkg(module_name: str, pip_name: str | None = None, required: bool = True) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        if not required:
            return False
        name = pip_name or module_name
        print(f"[deps] installing {name} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", name])
        importlib.import_module(module_name)
        return True


def _parse_tuple3(v: str | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(v, tuple):
        return v
    parts = [p.strip() for p in str(v).split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"patch size must be D,H,W (3 ints), got: {v}")
    out = tuple(int(x) for x in parts)
    if min(out) <= 0:
        raise ValueError(f"patch size must be positive, got: {out}")
    return out


def _load_nifti(path: str) -> np.ndarray:
    try:
        import SimpleITK as sitk  # type: ignore

        img = sitk.ReadImage(path)
        arr = np.asarray(sitk.GetArrayFromImage(img))
        return arr.astype(np.float32, copy=False)
    except Exception:
        import nibabel as nib  # type: ignore

        arr = np.asarray(nib.load(path).dataobj)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        return arr.astype(np.float32, copy=False)


def _crop_3d_with_padding(arr: np.ndarray, center_zyx: tuple[float, float, float], patch: tuple[int, int, int]) -> np.ndarray:
    dz, dy, dx = patch
    zc, yc, xc = [int(round(float(v))) for v in center_zyx]
    D, H, W = [int(v) for v in arr.shape]

    z0, y0, x0 = zc - dz // 2, yc - dy // 2, xc - dx // 2
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx

    sz0, sy0, sx0 = max(0, z0), max(0, y0), max(0, x0)
    sz1, sy1, sx1 = min(D, z1), min(H, y1), min(W, x1)
    crop = arr[sz0:sz1, sy0:sy1, sx0:sx1]

    pz0, py0, px0 = max(0, -z0), max(0, -y0), max(0, -x0)
    pz1, py1, px1 = max(0, z1 - D), max(0, y1 - H), max(0, x1 - W)
    if pz0 or pz1 or py0 or py1 or px0 or px1:
        fill = float(np.min(arr)) if arr.size else 0.0
        crop = np.pad(
            crop,
            ((pz0, pz1), (py0, py1), (px0, px1)),
            mode="constant",
            constant_values=fill,
        )

    if crop.shape != (dz, dy, dx):
        fill = float(np.min(crop)) if crop.size else 0.0
        out = np.full((dz, dy, dx), fill, dtype=np.float32)
        cz, cy, cx = min(dz, crop.shape[0]), min(dy, crop.shape[1]), min(dx, crop.shape[2])
        out[:cz, :cy, :cx] = crop[:cz, :cy, :cx]
        crop = out
    return crop.astype(np.float32, copy=False)


def _resize_3d(arr: np.ndarray, patch: tuple[int, int, int]) -> np.ndarray:
    if tuple(arr.shape) == tuple(patch):
        return arr.astype(np.float32, copy=False)
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    t = F.interpolate(t, size=patch, mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def _norm01(arr: np.ndarray) -> np.ndarray:
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32, copy=False)


def _roi_center_from_mask(mask: np.ndarray) -> tuple[float, float, float] | None:
    pos = np.argwhere(mask > 0)
    if pos.size == 0:
        return None
    c = pos.mean(axis=0)
    return float(c[0]), float(c[1]), float(c[2])


def _default_center(arr: np.ndarray) -> tuple[float, float, float]:
    D, H, W = arr.shape
    return (D - 1) / 2.0, (H - 1) / 2.0, (W - 1) / 2.0


def _write_csv(rows: list[dict[str, Any]], csv_path: Path, fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _build_text_index(meta_root: Path) -> dict[str, str]:
    """
    Build basename -> report text index from nearby json/jsonl/csv files.
    This is best-effort because dataset schemas vary.
    """
    idx: dict[str, str] = {}
    path_keys = ["image_path", "path", "file_path", "file", "nii_path", "ct_path", "image"]
    text_keys = ["answer", "finding", "findings", "report", "impression", "caption", "text"]

    def _pick(d: dict[str, Any], keys: list[str]) -> str:
        for k in keys:
            if k in d and d[k] is not None and str(d[k]).strip():
                return str(d[k]).strip()
        return ""

    for p in meta_root.rglob("*"):
        if not p.is_file():
            continue
        lp = p.suffix.lower()
        if lp not in {".json", ".jsonl", ".csv"}:
            continue
        try:
            if p.stat().st_size > 60 * 1024 * 1024:
                continue
        except Exception:
            continue
        try:
            if lp == ".json":
                obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                rows = obj if isinstance(obj, list) else (obj.get("data", []) if isinstance(obj, dict) else [])
                if not isinstance(rows, list):
                    continue
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    img = _pick(r, path_keys)
                    txt = _pick(r, text_keys)
                    if img and txt:
                        idx[Path(img).name] = txt
            elif lp == ".jsonl":
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        r = json.loads(line)
                        if not isinstance(r, dict):
                            continue
                        img = _pick(r, path_keys)
                        txt = _pick(r, text_keys)
                        if img and txt:
                            idx[Path(img).name] = txt
            else:
                import pandas as pd  # type: ignore

                df = pd.read_csv(p, encoding="utf-8", low_memory=False)
                pcol = next((c for c in path_keys if c in df.columns), None)
                tcol = next((c for c in text_keys if c in df.columns), None)
                if pcol and tcol:
                    for i in range(len(df)):
                        try:
                            img = str(df.iloc[i][pcol]).strip()
                            txt = str(df.iloc[i][tcol]).strip()
                        except Exception:
                            continue
                        if img and txt and img.lower() != "nan" and txt.lower() != "nan":
                            idx[Path(img).name] = txt
        except Exception:
            continue
    return idx


def _try_snapshot_download(repo_id: str, local_dir: Path, split_keyword: str | None) -> Path:
    from huggingface_hub import snapshot_download  # type: ignore

    allow_patterns = ["*.nii", "*.nii.gz", "*.json", "*.jsonl", "*.csv", "*.tsv"]
    if split_keyword:
        kw = split_keyword.strip().lower()
        allow_patterns += [f"*{kw}*/*", f"*{kw}*"]
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        token=os.environ.get("HF_TOKEN"),
        resume_download=True,
    )
    return local_dir


def _collect_nifti_files(root: Path, split_keyword: str | None = None) -> list[Path]:
    files = sorted(list(root.rglob("*.nii.gz")) + list(root.rglob("*.nii")))
    if not split_keyword:
        return files
    kw = split_keyword.lower()
    filt = [p for p in files if kw in str(p).lower()]
    return filt if filt else files


def _find_mask_candidate(img_path: Path) -> Path | None:
    stem = img_path.name
    cands = [
        stem.replace(".nii.gz", "_mask.nii.gz"),
        stem.replace(".nii.gz", "_seg.nii.gz"),
        stem.replace(".nii", "_mask.nii"),
        stem.replace(".nii", "_seg.nii"),
    ]
    # also try image->mask replacement in full path
    full = str(img_path)
    cands += [
        full.replace("/images/", "/masks/"),
        full.replace("\\images\\", "\\masks\\"),
    ]
    for c in cands:
        p = Path(c)
        if p.exists():
            return p
    return None


def prepare_rex(args: argparse.Namespace) -> Path:
    rex_dir = Path(args.rex_out_dir)
    img_out = rex_dir / "images"
    msk_out = rex_dir / "masks"
    raw_tmp = rex_dir / "_raw_download"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)
    raw_tmp.mkdir(parents=True, exist_ok=True)

    repo_candidates = [x.strip() for x in str(args.rex_repo_id).split(",") if x.strip()]
    dl_root = None
    for rid in repo_candidates:
        try:
            print(f"[rex] trying repo: {rid}", flush=True)
            dl_root = _try_snapshot_download(rid, raw_tmp / re.sub(r"[^a-zA-Z0-9_.-]", "_", rid), args.rex_split_keyword)
            break
        except Exception as e:
            print(f"[rex] repo failed: {rid} ({e})", flush=True)
    if dl_root is None:
        print("[rex] all repo candidates failed. Writing empty rex manifest and continue.", flush=True)
        csv_path = rex_dir / "rex_val.csv"
        _write_csv([], csv_path, ["image_path", "mask_path", "answer", "roi_center_3d"])
        return csv_path

    text_idx = _build_text_index(dl_root)
    nifti_files = _collect_nifti_files(dl_root, args.rex_split_keyword)
    if not nifti_files:
        raise RuntimeError(f"[rex] no NIfTI found under {dl_root}")

    rows: list[dict[str, Any]] = []
    patch = _parse_tuple3(args.patch_size_3d)
    limit = min(int(args.rex_limit), len(nifti_files))
    for i, nii in enumerate(nifti_files[:limit], start=1):
        mpath: Path | None = None
        try:
            arr = _load_nifti(str(nii))
            if arr.ndim != 3:
                continue
            mpath = _find_mask_candidate(nii)
            mask_arr = _load_nifti(str(mpath)) if mpath and mpath.exists() else None
            center = _roi_center_from_mask(mask_arr) if mask_arr is not None else None
            if center is None:
                center = _default_center(arr)

            crop = _crop_3d_with_padding(arr, center, patch)
            crop = _resize_3d(crop, patch)
            crop = _norm01(crop)
            roi_center = [patch[0] // 2, patch[1] // 2, patch[2] // 2]

            img_file = img_out / f"rex_{i:04d}.npz"
            np.savez_compressed(img_file, image=crop, roi_center_3d=np.asarray(roi_center, dtype=np.float32))

            out_mask = ""
            if mask_arr is not None:
                mask_crop = _crop_3d_with_padding(mask_arr.astype(np.float32), center, patch)
                mask_crop = _resize_3d(mask_crop, patch)
                mask_crop = (mask_crop > 0.5).astype(np.uint8)
                m_file = msk_out / f"rex_{i:04d}_mask.npz"
                np.savez_compressed(m_file, image=mask_crop)
                out_mask = str(m_file)

            answer = text_idx.get(nii.name, "") or "No finding text provided."
            rows.append(
                {
                    "image_path": str(img_file),
                    "mask_path": out_mask,
                    "answer": answer,
                    "roi_center_3d": ",".join(str(int(x)) for x in roi_center),
                }
            )
        finally:
            # Critical: free storage immediately.
            try:
                if nii.exists():
                    nii.unlink()
            except Exception:
                pass
            if mpath is not None:
                try:
                    if mpath.exists():
                        mpath.unlink()
                except Exception:
                    pass

    csv_path = rex_dir / "rex_val.csv"
    _write_csv(rows, csv_path, ["image_path", "mask_path", "answer", "roi_center_3d"])
    print(f"[rex] prepared {len(rows)} samples -> {csv_path}", flush=True)
    return csv_path


def _download_file(url: str, dst: Path) -> None:
    import requests  # type: ignore

    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dst.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _prepare_lidc_from_url_list(args: argparse.Namespace, patch: tuple[int, int, int]) -> list[dict[str, Any]]:
    url_list = Path(args.lidc_url_list)
    if not url_list.exists():
        raise FileNotFoundError(f"lidc url list not found: {url_list}")

    rows: list[dict[str, Any]] = []
    out_dir = Path(args.lidc_out_dir)
    img_out = out_dir / "images"
    msk_out = out_dir / "masks"
    tmp = out_dir / "_raw_download"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)

    parsed: list[tuple[str, str, str]] = []
    with url_list.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [x.strip() for x in s.split(",")]
            if len(parts) == 1:
                parsed.append((parts[0], "", ""))
            elif len(parts) == 2:
                parsed.append((parts[0], parts[1], ""))
            else:
                parsed.append((parts[0], parts[1], parts[2]))

    limit = min(int(args.lidc_limit), len(parsed))
    for i, (img_url, mask_url, grade) in enumerate(parsed[:limit], start=1):
        raw_img = tmp / f"lidc_{i:04d}.nii.gz"
        raw_msk = tmp / f"lidc_{i:04d}_mask.nii.gz"
        try:
            _download_file(img_url, raw_img)
            arr = _load_nifti(str(raw_img))
            mask_arr = None
            if mask_url:
                _download_file(mask_url, raw_msk)
                mask_arr = _load_nifti(str(raw_msk))

            center = _roi_center_from_mask(mask_arr) if mask_arr is not None else _default_center(arr)
            crop = _crop_3d_with_padding(arr, center, patch)
            crop = _resize_3d(crop, patch)
            crop = _norm01(crop)
            roi_center = [patch[0] // 2, patch[1] // 2, patch[2] // 2]

            img_file = img_out / f"lidc_{i:04d}.npz"
            np.savez_compressed(img_file, image=crop, roi_center_3d=np.asarray(roi_center, dtype=np.float32))

            out_mask = ""
            if mask_arr is not None:
                mask_crop = _crop_3d_with_padding(mask_arr.astype(np.float32), center, patch)
                mask_crop = _resize_3d(mask_crop, patch)
                mask_crop = (mask_crop > 0.5).astype(np.uint8)
                m_file = msk_out / f"lidc_{i:04d}_mask.npz"
                np.savez_compressed(m_file, image=mask_crop)
                out_mask = str(m_file)

            rows.append(
                {
                    "image_path": str(img_file),
                    "mask_path": out_mask,
                    "answer": "LIDC-IDRI nodule sample.",
                    "roi_center_3d": ",".join(str(int(x)) for x in roi_center),
                    "grade": grade if grade else "",
                }
            )
        finally:
            try:
                if raw_img.exists():
                    raw_img.unlink()
            except Exception:
                pass
            try:
                if raw_msk.exists():
                    raw_msk.unlink()
            except Exception:
                pass
    return rows


def _prepare_lidc_from_pylidc(args: argparse.Namespace, patch: tuple[int, int, int]) -> list[dict[str, Any]]:
    import pylidc as pl  # type: ignore
    from pylidc.utils import consensus  # type: ignore

    out_dir = Path(args.lidc_out_dir)
    img_out = out_dir / "images"
    msk_out = out_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    scans = list(pl.query(pl.Scan).all())
    random.shuffle(scans)
    rows: list[dict[str, Any]] = []
    for scan in scans:
        if len(rows) >= int(args.lidc_limit):
            break
        try:
            vol = scan.to_volume().astype(np.float32)
            if vol.ndim != 3:
                continue
            clusters = scan.cluster_annotations()
        except Exception:
            continue
        picked = None
        for c in clusters:
            diam = []
            for ann in c:
                try:
                    diam.append(float(getattr(ann, "diameter", 0.0)))
                except Exception:
                    continue
            if diam and max(diam) >= float(args.lidc_min_diameter_mm):
                picked = c
                break
        if picked is None:
            continue
        try:
            cmask, cbbox, _ = consensus(picked, clevel=0.5)
            full_mask = np.zeros_like(vol, dtype=np.uint8)
            full_mask[cbbox] = cmask.astype(np.uint8)
            center = _roi_center_from_mask(full_mask) or _default_center(vol)
            crop = _crop_3d_with_padding(vol, center, patch)
            crop = _resize_3d(crop, patch)
            crop = _norm01(crop)
            mask_crop = _crop_3d_with_padding(full_mask.astype(np.float32), center, patch)
            mask_crop = _resize_3d(mask_crop, patch)
            mask_crop = (mask_crop > 0.5).astype(np.uint8)
            roi_center = [patch[0] // 2, patch[1] // 2, patch[2] // 2]
            idx = len(rows) + 1
            img_file = img_out / f"lidc_{idx:04d}.npz"
            m_file = msk_out / f"lidc_{idx:04d}_mask.npz"
            np.savez_compressed(img_file, image=crop, roi_center_3d=np.asarray(roi_center, dtype=np.float32))
            np.savez_compressed(m_file, image=mask_crop)
            rows.append(
                {
                    "image_path": str(img_file),
                    "mask_path": str(m_file),
                    "answer": "LIDC-IDRI nodule sample.",
                    "roi_center_3d": ",".join(str(int(x)) for x in roi_center),
                    "grade": "",
                }
            )
        except Exception:
            continue
    return rows


def prepare_lidc(args: argparse.Namespace) -> Path:
    patch = _parse_tuple3(args.patch_size_3d)
    rows: list[dict[str, Any]] = []

    if args.lidc_url_list:
        print(f"[lidc] using URL list: {args.lidc_url_list}", flush=True)
        rows = _prepare_lidc_from_url_list(args, patch)
    else:
        has_pylidc = _ensure_pkg("pylidc", "pylidc", required=False)
        if has_pylidc:
            print("[lidc] using local pylidc database", flush=True)
            rows = _prepare_lidc_from_pylidc(args, patch)
        else:
            print("[lidc] pylidc not available and --lidc_url_list not set; writing empty manifest.", flush=True)

    out_dir = Path(args.lidc_out_dir)
    csv_path = out_dir / "lidc_val.csv"
    _write_csv(rows, csv_path, ["image_path", "mask_path", "answer", "roi_center_3d", "grade"])
    print(f"[lidc] prepared {len(rows)} samples -> {csv_path}", flush=True)
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare mini public ablation datasets (ReXGroundingCT + LIDC-IDRI).")
    parser.add_argument("--patch_size_3d", type=str, default="32,128,128")
    parser.add_argument("--rex_repo_id", type=str, default="surajpaib/ReXGroundingCT,rajpurkarlab/ReXGroundingCT,StanfordAIMI/ReXGroundingCT")
    parser.add_argument("--rex_split_keyword", type=str, default="val")
    parser.add_argument("--rex_limit", type=int, default=200)
    parser.add_argument("--rex_out_dir", type=str, default="/autodl-tmp/public_data/rex_val")
    parser.add_argument("--lidc_limit", type=int, default=200)
    parser.add_argument("--lidc_min_diameter_mm", type=float, default=3.0)
    parser.add_argument("--lidc_url_list", type=str, default="", help="Optional CSV/TXT list: image_url,mask_url,grade")
    parser.add_argument("--lidc_out_dir", type=str, default="/autodl-tmp/public_data/lidc_val")
    args = parser.parse_args()

    # Required dependencies
    _ensure_pkg("huggingface_hub", "huggingface_hub", required=True)
    _ensure_pkg("nibabel", "nibabel", required=True)
    _ensure_pkg("pandas", "pandas", required=True)
    _ensure_pkg("requests", "requests", required=True)
    _ensure_pkg("pylidc", "pylidc", required=False)

    Path(args.rex_out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.lidc_out_dir).mkdir(parents=True, exist_ok=True)

    rex_csv = prepare_rex(args)
    lidc_csv = prepare_lidc(args)

    summary = {
        "rex_csv": str(rex_csv),
        "lidc_csv": str(lidc_csv),
        "patch_size_3d": _parse_tuple3(args.patch_size_3d),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
