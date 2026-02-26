"""
Utilities for lung nodule contour visualization and basic nodule statistics.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

try:
    import SimpleITK as sitk
except Exception:
    sitk = None

try:
    import nibabel as nib
except Exception:
    nib = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4:
        return np.asarray(arr[..., 0])
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got ndim={arr.ndim}")
    return np.asarray(arr)


def _load_volume_zyx(path: str) -> tuple[np.ndarray, tuple[float, float, float], Any]:
    """
    Load volume in [z, y, x] axis order and return spacing in (dz, dy, dx).
    """
    if sitk is not None:
        img = sitk.ReadImage(str(path))
        arr = np.asarray(sitk.GetArrayFromImage(img))  # [z, y, x]
        arr = _ensure_3d(arr)
        sx, sy, sz = img.GetSpacing()[:3]  # SITK spacing order is (x, y, z)
        return arr.astype(np.float32), (float(sz), float(sy), float(sx)), img

    if nib is not None:
        img = nib.load(str(path))
        arr = np.asarray(img.dataobj)
        arr = _ensure_3d(arr)
        if arr.ndim == 3:
            arr = np.transpose(arr, (2, 1, 0))  # nibabel default -> [z, y, x]
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        if len(zooms) < 3:
            zooms = (1.0, 1.0, 1.0)
        dx, dy, dz = zooms
        return arr.astype(np.float32), (float(dz), float(dy), float(dx)), img

    raise ImportError("Need SimpleITK or nibabel to load NIfTI.")


def load_image_and_mask(
    image_path: str,
    mask_path: str,
    *,
    resample_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """
    Load image and mask in [z, y, x]. If SimpleITK is available, mask can be
    resampled to image geometry when geometry does not match.
    """
    if not Path(image_path).is_file():
        raise FileNotFoundError(
            f"图像文件不存在: {image_path}\n"
            "若使用 scripts/nodule_overlay_and_stats.py，请传入真实存在的 --image 路径（勿用占位符 xxx）。"
        )
    if not Path(mask_path).is_file():
        raise FileNotFoundError(
            f"Mask 文件不存在: {mask_path}\n"
            "请传入真实存在的 --mask 路径，通常为 nnunet_raw/.../labelsTr/病例ID.nii.gz。"
        )
    if sitk is not None:
        img_ref = sitk.ReadImage(str(image_path))
        msk_ref = sitk.ReadImage(str(mask_path))
        if resample_mask:
            geometry_diff = (
                msk_ref.GetSize() != img_ref.GetSize()
                or msk_ref.GetSpacing() != img_ref.GetSpacing()
                or msk_ref.GetDirection() != img_ref.GetDirection()
                or msk_ref.GetOrigin() != img_ref.GetOrigin()
            )
            if geometry_diff:
                msk_ref = sitk.Resample(
                    msk_ref,
                    img_ref,
                    sitk.Transform(),
                    sitk.sitkNearestNeighbor,
                    0,
                    sitk.sitkUInt8,
                )
        img = _ensure_3d(np.asarray(sitk.GetArrayFromImage(img_ref))).astype(np.float32)
        msk = _ensure_3d(np.asarray(sitk.GetArrayFromImage(msk_ref))).astype(np.uint8)
        if img.shape != msk.shape:
            raise ValueError(f"image shape {img.shape} != mask shape {msk.shape}")
        sx, sy, sz = img_ref.GetSpacing()[:3]
        spacing_zyx = (float(sz), float(sy), float(sx))
        return img, msk, spacing_zyx

    img, spacing_zyx, _ = _load_volume_zyx(image_path)
    msk, _, _ = _load_volume_zyx(mask_path)
    if img.shape != msk.shape:
        raise ValueError(f"image shape {img.shape} != mask shape {msk.shape}")
    return img.astype(np.float32), msk.astype(np.uint8), spacing_zyx


def find_best_slice(mask_3d: np.ndarray, slice_axis: int = 0) -> int:
    if mask_3d.ndim != 3:
        return 0
    binary = (mask_3d > 0).astype(np.uint8)
    if binary.sum() == 0:
        return mask_3d.shape[slice_axis] // 2
    if slice_axis == 0:
        scores = binary.sum(axis=(1, 2))
    elif slice_axis == 1:
        scores = binary.sum(axis=(0, 2))
    else:
        scores = binary.sum(axis=(0, 1))
    return int(np.argmax(scores))


def load_slice_with_optional_mask(
    image_path: str,
    *,
    mask_path: str | None = None,
    slice_axis: int = 0,
    slice_idx: int | None = None,
) -> np.ndarray:
    """
    Load one 2D image slice. If mask_path exists and slice_idx is None,
    choose the slice with max positive mask pixels.
    """
    img, _, _ = _load_volume_zyx(image_path)
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image for slice extraction, got shape={img.shape}")

    if slice_idx is None and mask_path and Path(mask_path).exists():
        _, msk, _ = load_image_and_mask(image_path, mask_path)
        slice_idx = find_best_slice(msk, slice_axis=slice_axis)

    if slice_idx is None:
        slice_idx = img.shape[slice_axis] // 2

    slice_idx = int(max(0, min(slice_idx, img.shape[slice_axis] - 1)))
    if slice_axis == 0:
        return img[slice_idx, :, :].astype(np.float32)
    if slice_axis == 1:
        return img[:, slice_idx, :].astype(np.float32)
    return img[:, :, slice_idx].astype(np.float32)


def compute_nodule_stats(
    mask_3d: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> list[dict[str, Any]]:
    """
    Connected-component nodule stats on binary mask (>0).
    """
    if ndi is None:
        raise ImportError("scipy is required for connected-component nodule stats.")
    binary = (np.asarray(mask_3d) > 0).astype(np.uint8)
    labeled, num = ndi.label(binary)
    if num <= 0:
        return []

    dz, dy, dx = spacing_zyx
    voxel_volume_mm3 = float(dz * dy * dx)
    rows: list[dict[str, Any]] = []
    for nid in range(1, num + 1):
        coords = np.argwhere(labeled == nid)
        if coords.size == 0:
            continue
        z_idx, y_idx, x_idx = coords[:, 0], coords[:, 1], coords[:, 2]
        voxel_count = int(coords.shape[0])
        volume_mm3 = voxel_count * voxel_volume_mm3
        eq_diameter_mm = float(((6.0 * volume_mm3) / np.pi) ** (1.0 / 3.0))

        rows.append(
            {
                "nodule_id": int(nid),
                "voxel_count": voxel_count,
                "volume_mm3": round(volume_mm3, 3),
                "equivalent_diameter_mm": round(eq_diameter_mm, 3),
                "center_z_mm": round(float(z_idx.mean() * dz), 3),
                "center_y_mm": round(float(y_idx.mean() * dy), 3),
                "center_x_mm": round(float(x_idx.mean() * dx), 3),
                "bbox_z_mm": round(float((z_idx.max() - z_idx.min() + 1) * dz), 3),
                "bbox_y_mm": round(float((y_idx.max() - y_idx.min() + 1) * dy), 3),
                "bbox_x_mm": round(float((x_idx.max() - x_idx.min() + 1) * dx), 3),
            }
        )

    rows.sort(key=lambda r: (r["volume_mm3"], r["nodule_id"]), reverse=True)
    return rows


def _write_stats_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "nodule_id",
        "voxel_count",
        "volume_mm3",
        "equivalent_diameter_mm",
        "center_z_mm",
        "center_y_mm",
        "center_x_mm",
        "bbox_z_mm",
        "bbox_y_mm",
        "bbox_x_mm",
    ]
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_contour_overlay_png(
    image_3d: np.ndarray,
    mask_3d: np.ndarray,
    out_png: Path,
    *,
    slice_index: int | None = None,
    line_width: float = 1.8,
    fill_alpha: float = 0.22,
) -> int:
    if plt is None:
        raise ImportError("matplotlib is required for contour visualization.")

    image_3d = _ensure_3d(np.asarray(image_3d))
    mask_3d = _ensure_3d(np.asarray(mask_3d))
    if image_3d.shape != mask_3d.shape:
        raise ValueError(f"image shape {image_3d.shape} != mask shape {mask_3d.shape}")

    z = find_best_slice(mask_3d, 0) if slice_index is None else int(slice_index)
    z = max(0, min(z, image_3d.shape[0] - 1))
    img = image_3d[z].astype(np.float32)
    msk = (mask_3d[z] > 0).astype(np.uint8)

    vmin, vmax = np.percentile(img, (1, 99))
    if float(vmax - vmin) < 1e-6:
        norm = np.zeros_like(img, dtype=np.float32)
    else:
        norm = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(norm, cmap="gray")
    if msk.any():
        ax.imshow(np.ma.masked_where(msk <= 0, msk), cmap="autumn", alpha=fill_alpha)
        ax.contour(msk, levels=[0.5], colors=["#ff3b30"], linewidths=[line_width])
    ax.set_title(f"z={z}", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return z


def generate_nodule_contour_outputs(
    image_path: str,
    mask_path: str,
    output_dir: str | Path,
    *,
    line_width: float = 1.8,
    fill_alpha: float = 0.22,
) -> dict[str, Any]:
    """
    End-to-end helper:
    - load image+mask
    - compute connected-component nodule stats
    - write nodules.csv
    - write contour overlay png
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_3d, mask_3d, spacing_zyx = load_image_and_mask(image_path, mask_path, resample_mask=True)
    rows = compute_nodule_stats(mask_3d, spacing_zyx)
    csv_path = out_dir / "nodules.csv"
    _write_stats_csv(rows, csv_path)

    png_path = out_dir / "overlay_contour.png"
    z_idx = save_contour_overlay_png(
        image_3d,
        mask_3d,
        png_path,
        line_width=line_width,
        fill_alpha=fill_alpha,
    )

    return {
        "nodule_count": int(len(rows)),
        "stats_csv": str(csv_path),
        "overlay_png": str(png_path),
        "best_slice_index": int(z_idx),
    }
