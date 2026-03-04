"""
Medical VLM dataset: load CT NIfTI + report CSV.
Supports optional mask_path for lesion-guided slice selection.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import nibabel as nib
    _HAS_NIBABEL = True
except ImportError:
    _HAS_NIBABEL = False

try:
    import SimpleITK as sitk
    _HAS_SITK = True
except ImportError:
    _HAS_SITK = False

CAPTION_DEFAULT_QUESTION_NO_NL = (
    "Please generate a chest CT report in four sections: "
    "Findings, Conclusion, Recommendation, and Pathology Tendency."
)
CAPTION_DEFAULT_QUESTION = CAPTION_DEFAULT_QUESTION_NO_NL + "\n"


def _load_nifti_volume(path: str):
    """Load NIfTI and return (image, array[z,y,x])."""
    if _HAS_SITK:
        img = sitk.ReadImage(path)
        arr = np.asarray(sitk.GetArrayFromImage(img))
        return img, arr
    if _HAS_NIBABEL:
        img = nib.load(path)
        arr = np.asarray(img.dataobj)
        return img, arr
    raise ImportError("Need SimpleITK or nibabel to load NIfTI.")


def _load_array_any(path: str) -> np.ndarray:
    """
    Load a 3D volume from multiple lightweight formats.
    Supported:
      - .nii / .nii.gz
      - .npz (expects key 'image' or first array)
      - .npy
      - .pt / .pth (expects tensor/ndarray or dict with 'image')
    """
    p = str(path)
    lp = p.lower()

    if lp.endswith(".npz"):
        obj = np.load(p, allow_pickle=False)
        if "image" in obj.files:
            arr = obj["image"]
        elif obj.files:
            arr = obj[obj.files[0]]
        else:
            raise ValueError(f"Empty npz file: {p}")
    elif lp.endswith(".npy"):
        arr = np.load(p, allow_pickle=False)
    elif lp.endswith(".pt") or lp.endswith(".pth"):
        payload = torch.load(p, map_location="cpu")
        if isinstance(payload, dict) and "image" in payload:
            payload = payload["image"]
        if torch.is_tensor(payload):
            arr = payload.detach().cpu().numpy()
        else:
            arr = np.asarray(payload)
    else:
        _, arr = _load_nifti_volume(p)

    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={arr.shape} from: {p}")
    return arr.astype(np.float32, copy=False)


def _load_roi_center_any(path: str) -> Optional[np.ndarray]:
    """
    Try reading roi_center_3d from lightweight files.
    Returns None if unavailable.
    """
    p = str(path)
    lp = p.lower()
    try:
        if lp.endswith(".npz"):
            obj = np.load(p, allow_pickle=False)
            if "roi_center_3d" in obj.files:
                c = np.asarray(obj["roi_center_3d"], dtype=np.float32).reshape(-1)
                if c.size >= 3:
                    return np.array([float(c[0]), float(c[1]), float(c[2])], dtype=np.float32)
        elif lp.endswith(".pt") or lp.endswith(".pth"):
            payload = torch.load(p, map_location="cpu")
            if isinstance(payload, dict) and "roi_center_3d" in payload:
                c = payload["roi_center_3d"]
                if torch.is_tensor(c):
                    c = c.detach().cpu().numpy()
                c = np.asarray(c, dtype=np.float32).reshape(-1)
                if c.size >= 3:
                    return np.array([float(c[0]), float(c[1]), float(c[2])], dtype=np.float32)
    except Exception:
        return None
    return None


def _best_slice_index_from_mask(mask_arr: np.ndarray, slice_axis: int) -> int:
    """Select slice with max nonzero voxels along axis."""
    if mask_arr.ndim != 3:
        return mask_arr.shape[slice_axis] // 2
    binary = (mask_arr > 0).astype(np.float64)
    if binary.sum() == 0:
        return mask_arr.shape[slice_axis] // 2
    if slice_axis == 0:
        slice_sums = binary.sum(axis=(1, 2))
    elif slice_axis == 1:
        slice_sums = binary.sum(axis=(0, 2))
    else:
        slice_sums = binary.sum(axis=(0, 1))
    return int(np.argmax(slice_sums))


def _center_of_mask_on_slice(mask_arr: np.ndarray, slice_axis: int, slice_idx: int) -> np.ndarray:
    """
    Compute lesion center (y, x) on a selected 2D slice.
    Returns [-1, -1] if the mask is empty/invalid on that slice.
    """
    invalid = np.array([-1.0, -1.0], dtype=np.float32)
    if mask_arr.ndim == 2:
        mask2d = mask_arr > 0
    elif mask_arr.ndim == 3:
        if slice_axis == 0:
            mask2d = mask_arr[slice_idx, :, :] > 0
        elif slice_axis == 1:
            mask2d = mask_arr[:, slice_idx, :] > 0
        else:
            mask2d = mask_arr[:, :, slice_idx] > 0
    else:
        return invalid

    ys, xs = np.where(mask2d)
    if ys.size == 0:
        return invalid
    return np.array([float(ys.mean()), float(xs.mean())], dtype=np.float32)


def _load_nifti_slice(
    path: str,
    slice_axis: int = 0,
    slice_idx: Optional[int] = None,
    mask_path: Optional[str] = None,
    return_roi: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Load one 2D slice; prefer lesion-guided slice when mask exists."""
    if _HAS_SITK:
        ct_img = sitk.ReadImage(path)
        arr = np.asarray(sitk.GetArrayFromImage(ct_img))
    elif _HAS_NIBABEL:
        ct_img = nib.load(path)
        arr = np.asarray(ct_img.dataobj)
    else:
        raise ImportError("Need SimpleITK or nibabel to load NIfTI.")

    if arr.ndim == 2:
        out2d = arr.astype(np.float32)
        roi_center = np.array([-1.0, -1.0], dtype=np.float32)
        if mask_path and Path(mask_path).exists():
            try:
                if _HAS_SITK:
                    m_img = sitk.ReadImage(mask_path)
                    m_arr = np.asarray(sitk.GetArrayFromImage(m_img))
                else:
                    _, m_arr = _load_nifti_volume(mask_path)
                roi_center = _center_of_mask_on_slice(m_arr, slice_axis=0, slice_idx=0)
            except Exception:
                roi_center = np.array([-1.0, -1.0], dtype=np.float32)
        return (out2d, roi_center) if return_roi else out2d

    if arr.ndim == 3:
        mask_arr = None
        if slice_idx is None and mask_path and Path(mask_path).exists():
            if _HAS_SITK:
                mask_img = sitk.ReadImage(mask_path)
                mask_bin = sitk.Cast(mask_img > 0, sitk.sitkUInt8)
                if (mask_bin.GetSize() != ct_img.GetSize() or
                    mask_bin.GetSpacing() != ct_img.GetSpacing() or
                    mask_bin.GetDirection() != ct_img.GetDirection() or
                    mask_bin.GetOrigin() != ct_img.GetOrigin()):
                    mask_bin = sitk.Resample(
                        mask_bin, ct_img, sitk.Transform(),
                        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
                    )
                mask_arr = np.asarray(sitk.GetArrayFromImage(mask_bin))
            else:
                _, mask_arr = _load_nifti_volume(mask_path)
            slice_idx = _best_slice_index_from_mask(mask_arr, slice_axis)

        if slice_idx is None:
            # No mask: use center slice for train/infer consistency
            d = arr.shape[slice_axis]
            slice_idx = d // 2

        if slice_axis == 0:
            out = arr[slice_idx, :, :]
        elif slice_axis == 1:
            out = arr[:, slice_idx, :]
        else:
            out = arr[:, :, slice_idx]

        roi_center = np.array([-1.0, -1.0], dtype=np.float32)
        if mask_path and Path(mask_path).exists():
            try:
                if mask_arr is None:
                    if _HAS_SITK:
                        mask_img = sitk.ReadImage(mask_path)
                        mask_bin = sitk.Cast(mask_img > 0, sitk.sitkUInt8)
                        if (mask_bin.GetSize() != ct_img.GetSize() or
                            mask_bin.GetSpacing() != ct_img.GetSpacing() or
                            mask_bin.GetDirection() != ct_img.GetDirection() or
                            mask_bin.GetOrigin() != ct_img.GetOrigin()):
                            mask_bin = sitk.Resample(
                                mask_bin, ct_img, sitk.Transform(),
                                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
                            )
                        mask_arr = np.asarray(sitk.GetArrayFromImage(mask_bin))
                    else:
                        _, mask_arr = _load_nifti_volume(mask_path)
                roi_center = _center_of_mask_on_slice(mask_arr, slice_axis=slice_axis, slice_idx=slice_idx)
            except Exception:
                roi_center = np.array([-1.0, -1.0], dtype=np.float32)

        out2d = out.astype(np.float32)
        return (out2d, roi_center) if return_roi else out2d

    raise ValueError(f"Unsupported NIfTI ndim={arr.ndim}")


def _crop_3d_with_padding(
    arr: np.ndarray,
    center_zyx: np.ndarray,
    patch_size_3d: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop [D,H,W] around center with target patch size.
    Pads with volume minimum value when out-of-bound.
    Returns:
      - crop: [Dz, Dy, Dx]
      - roi_center_in_crop: [z,y,x] inside returned crop
    """
    if arr.ndim != 3:
        raise ValueError(f"_crop_3d_with_padding expects 3D array, got ndim={arr.ndim}")

    dz, dy, dx = [int(v) for v in patch_size_3d]
    zc = int(round(float(center_zyx[0])))
    yc = int(round(float(center_zyx[1])))
    xc = int(round(float(center_zyx[2])))

    D, H, W = arr.shape
    z0, y0, x0 = zc - dz // 2, yc - dy // 2, xc - dx // 2
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx

    sz0, sy0, sx0 = max(0, z0), max(0, y0), max(0, x0)
    sz1, sy1, sx1 = min(D, z1), min(H, y1), min(W, x1)

    crop = arr[sz0:sz1, sy0:sy1, sx0:sx1]

    pz0, py0, px0 = max(0, -z0), max(0, -y0), max(0, -x0)
    pz1, py1, px1 = max(0, z1 - D), max(0, y1 - H), max(0, x1 - W)
    if pz0 or pz1 or py0 or py1 or px0 or px1:
        pad_value = float(arr.min()) if arr.size > 0 else 0.0
        crop = np.pad(
            crop,
            ((pz0, pz1), (py0, py1), (px0, px1)),
            mode="constant",
            constant_values=pad_value,
        )

    if crop.shape != (dz, dy, dx):
        # Final safety to strict target shape.
        out = np.full((dz, dy, dx), float(crop.min()) if crop.size > 0 else 0.0, dtype=np.float32)
        cz, cy, cx = min(dz, crop.shape[0]), min(dy, crop.shape[1]), min(dx, crop.shape[2])
        out[:cz, :cy, :cx] = crop[:cz, :cy, :cx]
        crop = out

    roi_center_in_crop = np.array(
        [
            float(np.clip(zc - sz0 + pz0, 0, dz - 1)),
            float(np.clip(yc - sy0 + py0, 0, dy - 1)),
            float(np.clip(xc - sx0 + px0, 0, dx - 1)),
        ],
        dtype=np.float32,
    )
    return crop.astype(np.float32), roi_center_in_crop


def _load_nifti_crop_3d(
    path: str,
    patch_size_3d: tuple[int, int, int] = (32, 128, 128),
    mask_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load NIfTI volume and crop a 3D patch around lesion center.
    - If mask exists and non-empty: use mask 3D center of mass (z,y,x).
    - Else fallback to volume center.
    Returns:
      - cropped volume [D,H,W]
      - roi_center_3d in cropped volume coords [z,y,x]
    """
    arr = _load_array_any(path)
    path_lower = str(path).lower()
    is_nifti = path_lower.endswith(".nii") or path_lower.endswith(".nii.gz")
    ct_img = None
    if is_nifti and _HAS_SITK:
        try:
            ct_img = sitk.ReadImage(path)
        except Exception:
            ct_img = None

    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported NIfTI ndim for 3D crop: {arr.ndim}")

    center = np.array([arr.shape[0] / 2.0, arr.shape[1] / 2.0, arr.shape[2] / 2.0], dtype=np.float32)

    # Prefer stored roi_center_3d when loading pre-cropped .npz/.pt assets.
    roi_from_file = _load_roi_center_any(path)
    if roi_from_file is not None and np.all(np.isfinite(roi_from_file)):
        center = roi_from_file.astype(np.float32)

    if mask_path and Path(mask_path).exists():
        try:
            if is_nifti and _HAS_SITK and ct_img is not None:
                mask_img = sitk.ReadImage(mask_path)
                mask_bin = sitk.Cast(mask_img > 0, sitk.sitkUInt8)
                if (mask_bin.GetSize() != ct_img.GetSize() or
                    mask_bin.GetSpacing() != ct_img.GetSpacing() or
                    mask_bin.GetDirection() != ct_img.GetDirection() or
                    mask_bin.GetOrigin() != ct_img.GetOrigin()):
                    mask_bin = sitk.Resample(
                        mask_bin, ct_img, sitk.Transform(),
                        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
                    )
                mask_arr = np.asarray(sitk.GetArrayFromImage(mask_bin))
            else:
                mask_arr = _load_array_any(mask_path)
            pos = np.argwhere(mask_arr > 0)
            if pos.size > 0:
                center = pos.mean(axis=0).astype(np.float32)
        except Exception:
            pass

    crop, roi_center_in_crop = _crop_3d_with_padding(arr.astype(np.float32), center, patch_size_3d)
    return crop, roi_center_in_crop


def _resize_to_patch(arr: np.ndarray, patch_size: int = 512) -> np.ndarray:
    """Resize 2D array to patch_size x patch_size (bilinear)."""
    if arr.shape[0] == patch_size and arr.shape[1] == patch_size:
        return arr
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(
        t,
        size=(patch_size, patch_size),
        mode="bilinear",
        align_corners=False,
    )
    return t.squeeze().numpy()


class MedicalVLMDataset(Dataset):
    """CSV with image_path, question, answer. Optional mask_path."""

    def __init__(
        self,
        csv_path: str | Path,
        prompt_json_file: Optional[str | Path] = None,
        patch_size: int = 512,
        patch_size_3d: tuple[int, int, int] = (32, 128, 128),
        spatial_dims: int = 2,
        slice_axis: int = 0,
        normalize: bool = True,
        image_root: Optional[str | Path] = None,
        mask_root: Optional[str | Path] = None,
    ):
        import pandas as pd
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.image_root = str(image_root) if image_root else None
        self.mask_root = str(mask_root) if mask_root else None

        if "image_path" in self.df.columns:
            self.img_paths = self.df["image_path"].astype(str).tolist()
            if __import__("sys").platform == "linux":
                self.img_paths = [_win_to_wsl_path(x) for x in self.img_paths]
        else:
            id_col = "image_id" if "image_id" in self.df.columns else ("搴忓彿" if "搴忓彿" in self.df.columns else None)
            if id_col is None:
                raise ValueError("CSV must contain `image_path` or one of (`image_id`, `序号`).")
            if not self.image_root:
                raise ValueError("When CSV has no `image_path`, `image_root` must be provided.")
            ids = self.df[id_col].astype(int).tolist()
            self.img_paths = [str(Path(self.image_root) / f"{i}.nii.gz") for i in ids]

        if "mask_path" in self.df.columns:
            self.mask_paths = self.df["mask_path"].astype(str).tolist()
            if __import__("sys").platform == "linux":
                self.mask_paths = [_win_to_wsl_path(x) for x in self.mask_paths]
        elif self.mask_root:
            self.mask_paths = [str(Path(self.mask_root) / Path(p).name) for p in self.img_paths]
        else:
            self.mask_paths = [None] * len(self.img_paths)

        self.questions = (
            self.df["question"].astype(str).tolist()
            if "question" in self.df.columns
            else [""] * len(self.img_paths)
        )
        self.answers = self.df["answer"].astype(str).tolist()
        # Optional invasive-grade label. Accept any of:
        #   grade (0/1/2/3), grade_id (0/1/2/3), grade_text (AAH/AIS/MIA/IAC)
        # Invalid/missing values are normalized to -1.
        grade_text_map = {"AAH": 0, "AIS": 1, "MIA": 2, "IAC": 3}
        n = len(self.img_paths)
        grades = [-1] * n

        grade_series = self.df["grade"] if "grade" in self.df.columns else None
        grade_id_series = self.df["grade_id"] if "grade_id" in self.df.columns else None
        grade_text_series = self.df["grade_text"] if "grade_text" in self.df.columns else None

        for i in range(n):
            g = -1
            if grade_series is not None:
                try:
                    v = int(float(grade_series.iloc[i]))
                    if 0 <= v <= 3:
                        g = v
                except Exception:
                    pass
            if g < 0 and grade_id_series is not None:
                try:
                    v = int(float(grade_id_series.iloc[i]))
                    if 0 <= v <= 3:
                        g = v
                except Exception:
                    pass
            if g < 0 and grade_text_series is not None:
                t = str(grade_text_series.iloc[i]).strip().upper()
                if t in grade_text_map:
                    g = grade_text_map[t]
            grades[i] = g

        self.grades = grades
        self.patch_size = patch_size
        self.patch_size_3d = tuple(int(v) for v in patch_size_3d)
        self.spatial_dims = int(spatial_dims)
        if self.spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {self.spatial_dims}")
        self.slice_axis = slice_axis
        self.normalize = normalize
        self.prompts = []
        if prompt_json_file and Path(prompt_json_file).exists():
            with open(prompt_json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.prompts = data.get("caption_prompt", data.get("prompts", []))

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.img_paths[index]
        question = self.questions[index] if self.questions[index] else (
            random.choice(self.prompts) if self.prompts else CAPTION_DEFAULT_QUESTION_NO_NL
        )
        answer = self.answers[index]
        mask_path = self.mask_paths[index] if index < len(self.mask_paths) else None
        if mask_path and not Path(mask_path).exists():
            mask_path = None
        roi_center = np.array([-1.0, -1.0], dtype=np.float32)
        roi_center_3d = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        try:
            if self.spatial_dims == 3:
                vol, roi_center_3d = _load_nifti_crop_3d(
                    path,
                    patch_size_3d=self.patch_size_3d,
                    mask_path=mask_path,
                )
                if index == 0:
                    print(f"\n[DEBUG Dataset] Loading 3D: {Path(path).name}", flush=True)
                    print(f"               Shape: {vol.shape} (D,H,W)", flush=True)
                if self.normalize:
                    mn, mx = vol.min(), vol.max()
                    if mx - mn > 1e-8:
                        vol = (vol - mn) / (mx - mn)
                    else:
                        vol = np.zeros_like(vol)
                image = torch.from_numpy(vol).float().unsqueeze(0)  # [1, D, H, W]
            else:
                arr, roi_center = _load_nifti_slice(
                    path,
                    self.slice_axis,
                    None,
                    mask_path=mask_path,
                    return_roi=True,
                )
                if index == 0:
                    print(f"\n[DEBUG Dataset] Loading: {Path(path).name}", flush=True)
                    print(f"               Shape: {arr.shape} (2D slice, axis={self.slice_axis})", flush=True)
                    if arr.size > 0 and float(arr.max() - arr.min()) < 1e-6:
                        print("               [WARNING] Image is completely black/empty! Consider slice selection.", flush=True)

                h0, w0 = int(arr.shape[0]), int(arr.shape[1])
                arr = _resize_to_patch(arr, self.patch_size)
                # ROI center is kept in resized image coordinate convention.
                if roi_center is None or len(roi_center) != 2:
                    roi_center = np.array([-1.0, -1.0], dtype=np.float32)
                else:
                    roi_center = np.asarray(roi_center, dtype=np.float32)
                if roi_center[0] >= 0 and roi_center[1] >= 0:
                    sy = float(self.patch_size) / max(float(h0), 1.0)
                    sx = float(self.patch_size) / max(float(w0), 1.0)
                    roi_center[0] = float(np.clip(roi_center[0] * sy, 0.0, self.patch_size - 1.0))
                    roi_center[1] = float(np.clip(roi_center[1] * sx, 0.0, self.patch_size - 1.0))
                else:
                    roi_center = np.array([-1.0, -1.0], dtype=np.float32)
                if self.normalize:
                    mn, mx = arr.min(), arr.max()
                    if mx - mn > 1e-8:
                        arr = (arr - mn) / (mx - mn)
                    else:
                        arr = np.zeros_like(arr)
                image = torch.from_numpy(arr).float().unsqueeze(0)  # [1, H, W]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image at index={index}: {path}. "
                "Please fix the CSV/image paths before training."
            ) from e
        grade = self.grades[index] if index < len(self.grades) else -1
        return {
            "image": image,
            "question": question,
            "answer": answer,
            "image_path": path,
            # DataLoader default_collate cannot batch None values.
            "mask_path": mask_path if mask_path is not None else "",
            "roi_center": torch.tensor(roi_center, dtype=torch.float32),
            "roi_center_3d": torch.tensor(roi_center_3d, dtype=torch.float32),
            "grade": grade,
        }


def _win_to_wsl_path(p: str) -> str:
    import sys
    if sys.platform != "linux":
        return p
    p = p.replace("\\", "/").strip()
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        return f"/mnt/{drive}" + p[2:]
    if p.startswith("d:/") or p.startswith("D:/"):
        return "/mnt/d/" + p[3:]
    if p.startswith("c:/") or p.startswith("C:/"):
        return "/mnt/c/" + p[3:]
    return p


def _wsl_path(p: str) -> str:
    return _win_to_wsl_path(p)


def load_paths_config(config_path: Optional[str | Path] = None) -> dict:
    default = {
        "nnunet_raw": "d:/nnunet_raw",
        "nnunet_preprocessed": "d:/nnunet_preprocessed",
        "nnunet_results": "d:/nnunet_results",
        "caption_csv_train": "d:/unn-net/train_radfm_315.csv",
        "caption_csv_val": "d:/unn-net/val_radfm_315.csv",
        "caption_prompt_json": "d:/unn-net/radfm_caption_prompt.json",
        "spatial_dims": 2,
        "patch_size_3d": [32, 128, 128],
    }
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "paths.yaml"
    path = Path(config_path)
    if not path.exists():
        out = {**default}
    else:
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                out = {**default, **yaml.safe_load(f)}
        except Exception:
            out = {**default}
    if __import__("sys").platform == "linux":
        for k, v in list(out.items()):
            if isinstance(v, str) and (v.startswith("d:/") or v.startswith("D:/") or v.startswith("d:\\") or v.startswith("D:\\")):
                out[k] = _wsl_path(v)
    return out

