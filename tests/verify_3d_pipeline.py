from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data import medical_vlm_dataset as ds_mod
from model.forward_medical_vlm import MedicalVLM


def _device_for_vim() -> torch.device:
    # mamba_ssm/causal_conv1d kernel requires CUDA runtime.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def verify_dataset_mock() -> None:
    print("[1/3] Verify dataset 3D output...")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        img = root / "dummy_ct.nii.gz"
        msk = root / "dummy_mask.nii.gz"
        img.touch()
        msk.touch()
        csv_path = root / "mock.csv"
        pd.DataFrame(
            [
                {
                    "image_path": str(img),
                    "mask_path": str(msk),
                    "question": "Q",
                    "answer": "A",
                    "grade": 3,
                }
            ]
        ).to_csv(csv_path, index=False, encoding="utf-8-sig")

        original_loader = ds_mod._load_nifti_crop_3d

        def _fake_load_nifti_crop_3d(path, patch_size_3d=(32, 128, 128), mask_path=None):
            dz, dy, dx = patch_size_3d
            vol = np.zeros((dz, dy, dx), dtype=np.float32)
            vol[dz // 2, dy // 2, dx // 2] = 1.0
            roi = np.array([dz // 2, dy // 2, dx // 2], dtype=np.float32)
            return vol, roi

        ds_mod._load_nifti_crop_3d = _fake_load_nifti_crop_3d
        try:
            ds = ds_mod.MedicalVLMDataset(
                csv_path,
                spatial_dims=3,
                patch_size_3d=(32, 128, 128),
                normalize=False,
            )
            sample = ds[0]
            assert "roi_center_3d" in sample, "roi_center_3d missing in sample dict"
            assert sample["roi_center_3d"].shape == (3,), f"roi_center_3d shape error: {sample['roi_center_3d'].shape}"
            assert sample["image"].shape == (1, 32, 128, 128), f"image shape error: {sample['image'].shape}"
            print("  roi_center_3d shape:", tuple(sample["roi_center_3d"].shape))
            print("  image shape:", tuple(sample["image"].shape))
        finally:
            ds_mod._load_nifti_crop_3d = original_loader


def verify_model_forward_3d() -> None:
    print("[2/3] Verify MedicalVLM 3D forward...")
    torch.manual_seed(0)
    device = _device_for_vim()
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for VimBridge verification (mamba_ssm kernel).")

    image = torch.zeros(1, 1, 32, 128, 128, dtype=torch.float32, device=device)
    image[0, 0, 16, 64, 64] = 10.0
    roi_center = torch.tensor([[16.0, 64.0, 64.0]], dtype=torch.float32, device=device)

    # Keep total tokens at 164 to match experiment setting target.
    g3d = (4, 4, 4)   # 64 global tokens
    l3d = (2, 5, 10)  # 100 local tokens
    expected_tokens = int(np.prod(g3d) + np.prod(l3d))

    model = MedicalVLM(
        spatial_dims=3,
        vision_bridge_type="vim",
        bridge_d_model=128,
        global_pool_size_3d=g3d,
        local_crop_size_3d=l3d,
        ablation_mode="full",
    ).to(device).eval()

    with torch.no_grad():
        out_roi = model(image, roi_center_3d=roi_center)
        out_far = model(image, roi_center_3d=torch.tensor([[8.0, 20.0, 20.0]], dtype=torch.float32, device=device))

    assert out_roi.ndim == 3, f"unexpected output ndim={out_roi.ndim}"
    assert out_roi.shape[0] == 1, f"unexpected batch dim: {out_roi.shape}"
    assert out_roi.shape[1] == expected_tokens, f"unexpected token length: {out_roi.shape[1]} != {expected_tokens}"
    assert out_roi.shape[2] == 128, f"unexpected channel dim: {out_roi.shape[2]}"
    # Dynamic ROI check: changing crop center should change local stream tokens.
    assert not torch.allclose(out_roi, out_far), "dynamic 3D ROI crop not taking effect"
    print("  output shape:", tuple(out_roi.shape))


def verify_transformer_bridge_3d() -> None:
    print("[3/3] Verify Transformer bridge 3D forward...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.randn(1, 1, 32, 128, 128, dtype=torch.float32, device=device)
    roi_center = torch.tensor([[16.0, 64.0, 64.0]], dtype=torch.float32, device=device)

    g3d = (4, 4, 4)
    l3d = (2, 5, 10)
    expected_tokens = int(np.prod(g3d) + np.prod(l3d))

    model = MedicalVLM(
        spatial_dims=3,
        vision_bridge_type="transformer",
        bridge_d_model=128,
        global_pool_size_3d=g3d,
        local_crop_size_3d=l3d,
        ablation_mode="full",
    ).to(device).eval()

    with torch.no_grad():
        out = model(image, roi_center_3d=roi_center)

    assert out.shape[0] == 1 and out.shape[1] == expected_tokens and out.shape[2] == 128, (
        f"unexpected transformer output shape: {tuple(out.shape)}"
    )
    print("  Transformer Bridge Forward Pass: OK")
    print("  output shape:", tuple(out.shape))


if __name__ == "__main__":
    verify_dataset_mock()
    verify_model_forward_3d()
    verify_transformer_bridge_3d()
    print("3D PIPELINE SUCCESS")
