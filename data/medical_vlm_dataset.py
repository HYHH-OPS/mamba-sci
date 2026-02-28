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
    "请基于胸部CT仅按以下固定模板输出，禁止输出模板外内容。"
    "所见：<影像学所见，含定位/大小/征象>；"
    "结论：<1-3条结论>；"
    "建议：<随访或检查建议>；"
    "病理倾向：<炎性/肿瘤性/待定及理由>。"
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


def _load_nifti_slice(
    path: str,
    slice_axis: int = 0,
    slice_idx: Optional[int] = None,
    mask_path: Optional[str] = None,
) -> np.ndarray:
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
        return arr.astype(np.float32)

    if arr.ndim == 3:
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
        return out.astype(np.float32)

    raise ValueError(f"Unsupported NIfTI ndim={arr.ndim}")


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
            id_col = "image_id" if "image_id" in self.df.columns else ("序号" if "序号" in self.df.columns else None)
            if id_col is None:
                raise ValueError("CSV 缺少 image_path 且无 image_id/序号 列。")
            if not self.image_root:
                raise ValueError("CSV 无 image_path 时需提供 image_root。")
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
        # 可选：侵润/分级标签（AAH/AIS/MIA/IAC 等），若不存在则全部置为 -1
        if "grade" in self.df.columns:
            try:
                self.grades = self.df["grade"].astype(int).tolist()
            except Exception:
                # 若无法直接转为 int，则保持原始字符串，后续由训练侧自行解析
                self.grades = self.df["grade"].astype(str).tolist()
        else:
            self.grades = [-1] * len(self.img_paths)
        self.patch_size = patch_size
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
        try:
            arr = _load_nifti_slice(path, self.slice_axis, None, mask_path=mask_path)
            if index == 0:
                print(f"\n[DEBUG Dataset] Loading: {Path(path).name}", flush=True)
                print(f"               Shape: {arr.shape} (2D slice, axis={self.slice_axis})", flush=True)
                if arr.size > 0 and float(arr.max() - arr.min()) < 1e-6:
                    print("               [WARNING] Image is completely black/empty! Consider slice selection.", flush=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image at index={index}: {path}. "
                "Please fix the CSV/image paths before training."
            ) from e

        arr = _resize_to_patch(arr, self.patch_size)
        if self.normalize:
            mn, mx = arr.min(), arr.max()
            if mx - mn > 1e-8:
                arr = (arr - mn) / (mx - mn)
            else:
                arr = np.zeros_like(arr)
        image = torch.from_numpy(arr).float().unsqueeze(0)
        grade = self.grades[index] if index < len(self.grades) else -1
        return {
            "image": image,
            "question": question,
            "answer": answer,
            "image_path": path,
            # DataLoader default_collate cannot batch None values.
            "mask_path": mask_path if mask_path is not None else "",
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
