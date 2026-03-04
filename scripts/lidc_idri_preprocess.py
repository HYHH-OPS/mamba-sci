#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIDC-IDRI 结节预处理脚本（用于 Mamba 肺结节多模态 SCI）。
功能：筛选 ≥3 位医生标注的结节，按 BBox 裁剪 2.5D 切片，保存 224×224 及 dataset_index.csv。
依赖：pylidc，且需配置 ~/.pylidcrc 指向 LIDC-IDRI 的 DICOM 根目录。

环境配置简要说明：
  1) 安装： pip install pylidc
  2) 从 TCIA 下载 LIDC-IDRI 数据后，在本地建目录，例如 /data/LIDC-IDRI/，
     其下为按 PatientID 命名的子目录：LIDC-IDRI-0001, LIDC-IDRI-0002, ...，
     每个子目录内为该病例的 DICOM 文件（TCIA 下载可能带一层 uid 子目录，pylidc 会递归查找）。
  3) 配置文件 ~/.pylidcrc（Linux/Mac）或 C:\\Users\\[User]\\pylidc.conf（Windows），内容示例：
       [dicom]
       path = /data/LIDC-IDRI
       warn = True
  4) 无需单独配置 TCIA 账号到 pylidc；数据需先在 TCIA 网站下载到本地，path 指向该本地根目录即可。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# 尝试导入 pylidc，失败时给出配置提示
try:
    import pylidc as pl
except ImportError as e:
    print("未安装 pylidc，请执行: pip install pylidc", file=sys.stderr)
    raise SystemExit(1) from e


# 结节 ROI 外扩像素数（上下文）
DEFAULT_PAD_PIXELS = 15
# 2.5D 切片数（中心 ±1 共 3 张）
NUM_SLICES_2_5D = 3
# 输出切片尺寸（高分辨率特征图 28x28/32x32 的输入常用 224）
TARGET_SIZE = 224


def _resize_slice(slice_2d: np.ndarray, target: int = TARGET_SIZE) -> np.ndarray:
    """将 2D 切片 Resize 到 target x target（双线性插值）。"""
    if slice_2d.shape[0] == target and slice_2d.shape[1] == target:
        return slice_2d.astype(np.float32)
    try:
        from scipy.ndimage import zoom
        h, w = slice_2d.shape
        zoom_h, zoom_w = target / h, target / w
        out = zoom(slice_2d.astype(np.float32), (zoom_h, zoom_w), order=1)
        return out.astype(np.float32)
    except ImportError:
        # 无 scipy 时用简单最近邻
        from PIL import Image
        img = Image.fromarray(slice_2d.astype(np.float32))
        out = np.array(img.resize((target, target), Image.BILINEAR), dtype=np.float32)
        return out


def _malignancy_label(annotations: list) -> float:
    """多位医生恶性程度评分的平均值（1–5）。"""
    vals = [ann.malignancy for ann in annotations if hasattr(ann, "malignancy") and ann.malignancy is not None]
    if not vals:
        return 3.0
    return float(np.mean(vals))


def _texture_label(annotations: list) -> float:
    """多位医生纹理评分的平均值（1–5，用于区分磨玻璃/实性）。"""
    vals = [ann.texture for ann in annotations if hasattr(ann, "texture") and ann.texture is not None]
    if not vals:
        return 3.0
    return float(np.mean(vals))


def _crop_2_5d_from_volume(
    vol: np.ndarray,
    bbox_slices: tuple,
    center_k: int,
    num_slices: int = NUM_SLICES_2_5D,
    target_size: int = TARGET_SIZE,
) -> np.ndarray:
    """
    从 3D 体积中按 bbox 的 i,j 范围与 center_k 附近取 num_slices 张 2D 切片，堆叠为 2.5D，并 Resize 到 target_size。
    返回形状 (num_slices, target_size, target_size)，dtype float32。
    """
    si, sj, sk = bbox_slices
    half = num_slices // 2
    slices_list = []
    for i in range(num_slices):
        k = center_k - half + i
        k_use = max(sk.start, min(sk.stop - 1, k))
        slice_2d = np.asarray(vol[si, sj, k_use], dtype=np.float32)
        slice_2d = _resize_slice(slice_2d, target_size)
        slices_list.append(slice_2d)
    return np.stack(slices_list, axis=0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LIDC-IDRI 结节筛选、裁剪 2.5D、生成 dataset_index.csv（SCI 用）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/root/autodl-tmp/mamba-res/lidc_idri_2p5d",
        help="输出根目录：其下生成 nodules/ 与 dataset_index.csv",
    )
    parser.add_argument(
        "--pad_pixels",
        type=int,
        default=DEFAULT_PAD_PIXELS,
        help="BBox 外扩像素数（上下文）",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=TARGET_SIZE,
        help="输出切片 Resize 尺寸",
    )
    parser.add_argument(
        "--min_annotations",
        type=int,
        default=3,
        help="至少几位医生标注的结节才保留",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        choices=["npy", "png"],
        default="npy",
        help="保存格式：npy 单文件 2.5D 块，png 为每例 3 张图",
    )
    parser.add_argument(
        "--max_scans",
        type=int,
        default=None,
        help="最多处理多少个 Scan（用于调试，默认不限制）",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    nodules_dir = out_dir / "nodules"
    nodules_dir.mkdir(parents=True, exist_ok=True)

    # 查询所有 Scan（需已配置 ~/.pylidcrc 且 DICOM 路径正确）
    try:
        scans = pl.query(pl.Scan).all()
    except Exception as e:
        print(f"查询 LIDC Scan 失败，请检查 ~/.pylidcrc 与数据库。错误: {e}", file=sys.stderr)
        return 1

    if args.max_scans is not None:
        scans = scans[: args.max_scans]

    rows: list[dict] = []
    global_index = 0

    for scan in tqdm(scans, desc="Scan"):
        try:
            vol = scan.to_volume(verbose=False)
        except Exception as e:
            tqdm.write(f"跳过 Scan {getattr(scan, 'patient_id', scan)}: 加载 DICOM 失败 - {e}")
            continue

        try:
            nodules = scan.cluster_annotations(verbose=False)
        except Exception as e:
            tqdm.write(f"跳过 Scan {getattr(scan, 'patient_id', scan)}: cluster_annotations 失败 - {e}")
            continue

        for nodule_idx, ann_list in enumerate(nodules):
            if len(ann_list) < args.min_annotations:
                continue

            # 恶性程度与纹理：多位医生平均
            malignancy_score = round(_malignancy_label(ann_list), 2)
            texture_score = round(_texture_label(ann_list), 2)

            # 取第一个标注的 bbox 做裁剪（外扩 pad_pixels）
            ann0 = ann_list[0]
            try:
                bbox = ann0.bbox(pad=args.pad_pixels)
            except Exception as e:
                tqdm.write(f"跳过 nodule bbox: {e}")
                continue

            # 中心切片索引（用于 2.5D）
            center_k = int(round(ann0.centroid[2]))

            try:
                crop_3d = vol[bbox]
            except Exception as e:
                tqdm.write(f"跳过 crop: {e}")
                continue

            # 2.5D：3 张连续 2D 切片，Resize 到 target_size
            try:
                stack_2_5d = _crop_2_5d_from_volume(
                    vol, bbox, center_k,
                    num_slices=NUM_SLICES_2_5D,
                    target_size=args.target_size,
                )
            except Exception as e:
                tqdm.write(f"跳过 2.5D 生成: {e}")
                continue

            # 保存
            pid = getattr(scan, "patient_id", "unknown").replace(" ", "_")
            base_name = f"{pid}_nod{nodule_idx:03d}_idx{global_index:05d}"
            global_index += 1

            if args.save_format == "npy":
                image_path = nodules_dir / f"{base_name}.npy"
                np.save(image_path, stack_2_5d)
                path_for_csv = str(image_path.resolve())
            else:
                image_dir = nodules_dir / base_name
                image_dir.mkdir(parents=True, exist_ok=True)
                try:
                    from PIL import Image
                except ImportError:
                    tqdm.write("保存 png 需要 Pillow: pip install Pillow")
                    continue
                for s in range(stack_2_5d.shape[0]):
                    img = stack_2_5d[s].copy()
                    mi, mx = img.min(), img.max()
                    if mx > mi:
                        img = (img - mi) / (mx - mi) * 255
                    else:
                        img = np.zeros_like(img)
                    Image.fromarray(img.astype(np.uint8)).save(image_dir / f"slice_{s}.png")
                path_for_csv = str(image_dir.resolve())

            rows.append({
                "image_path": path_for_csv,
                "malignancy_score": malignancy_score,
                "texture": texture_score,
                "patient_id": pid,
                "nodule_idx": nodule_idx,
                "n_annotations": len(ann_list),
            })

    # 生成 dataset_index.csv
    index_path = out_dir / "dataset_index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "malignancy_score", "texture", "patient_id", "nodule_idx", "n_annotations"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"完成。共 {len(rows)} 条结节，索引表: {index_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
