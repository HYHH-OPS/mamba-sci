"""
Medical VLM visual forward.

Supports:
- 2D pipeline: [B, 1, H, W]
- 3D pipeline: [B, 1, D, H, W]
- Global-Local Feature Preserving Bridge (GL-FPB)
- Bridge type switch: vim / transformer
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Project modules (run from repo root or include repo in PYTHONPATH)
try:
    from vision import build_nnunet_encoder_light, build_nnunet_encoder_3d
    from bridge import VimBridge, TransformerBridge, CMIConnector
except ImportError:
    import sys

    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from vision import build_nnunet_encoder_light, build_nnunet_encoder_3d
    from bridge import VimBridge, TransformerBridge, CMIConnector


def _valid_roi_2d(rc: torch.Tensor) -> bool:
    return bool(torch.isfinite(rc).all() and rc[0] >= 0 and rc[1] >= 0)


def _valid_roi_3d(rc: torch.Tensor) -> bool:
    return bool(torch.isfinite(rc).all() and rc[0] >= 0 and rc[1] >= 0 and rc[2] >= 0)


def _as_tuple3(v, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if v is None:
        return tuple(int(x) for x in default)
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(",") if p.strip()]
        if len(parts) == 3:
            try:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
            except Exception:
                return tuple(int(x) for x in default)
        return tuple(int(x) for x in default)
    try:
        vals = tuple(int(x) for x in v)
        if len(vals) == 3:
            return vals
    except Exception:
        pass
    return tuple(int(x) for x in default)


def _roi_crop_2d(
    feat: torch.Tensor,
    size: int,
    roi_center: Optional[torch.Tensor] = None,
    image_hw: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """
    ROI crop [B,C,Hf,Wf] -> [B,C,size,size].
    roi_center is expected in input image coordinates [B,2]=(y,x).
    Invalid roi_center falls back to center crop.
    """
    if feat.dim() != 4:
        raise ValueError(f"_roi_crop_2d expects [B,C,H,W], got {tuple(feat.shape)}")
    bsz, _, h, w = feat.shape
    size = int(size)
    if size <= 0:
        return feat

    if roi_center is None:
        roi_center = torch.full((bsz, 2), -1.0, device=feat.device, dtype=torch.float32)
    else:
        roi_center = torch.as_tensor(roi_center, device=feat.device, dtype=torch.float32)
        if roi_center.ndim == 1:
            roi_center = roi_center.unsqueeze(0)
        if roi_center.shape[0] == 1 and bsz > 1:
            roi_center = roi_center.expand(bsz, -1)
        if roi_center.shape[0] != bsz or roi_center.shape[1] != 2:
            roi_center = torch.full((bsz, 2), -1.0, device=feat.device, dtype=torch.float32)

    img_h, img_w = image_hw if image_hw is not None else (h, w)
    out_list = []
    for i in range(bsz):
        rc = roi_center[i]
        if _valid_roi_2d(rc):
            # Map input-image coords -> feature-map coords.
            cy = float(rc[0].item()) * (float(h) / max(float(img_h), 1.0))
            cx = float(rc[1].item()) * (float(w) / max(float(img_w), 1.0))
        else:
            cy = (h - 1) / 2.0
            cx = (w - 1) / 2.0
        cy, cx = int(round(cy)), int(round(cx))

        top = cy - size // 2
        left = cx - size // 2
        bottom = top + size
        right = left + size

        st, sl = max(0, top), max(0, left)
        sb, sr = min(h, bottom), min(w, right)
        crop = feat[i : i + 1, :, st:sb, sl:sr]

        pt, pb = max(0, -top), max(0, bottom - h)
        pl, pr = max(0, -left), max(0, right - w)
        if pt or pb or pl or pr:
            crop = F.pad(crop, (pl, pr, pt, pb), mode="replicate")
        if crop.shape[-2:] != (size, size):
            crop = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)
        out_list.append(crop)
    return torch.cat(out_list, dim=0)


def _roi_crop_3d(
    feat: torch.Tensor,
    size_zyx: tuple[int, int, int],
    roi_center_3d: Optional[torch.Tensor] = None,
    image_dhw: Optional[tuple[int, int, int]] = None,
) -> torch.Tensor:
    """
    ROI crop [B,C,Df,Hf,Wf] -> [B,C,zd,zh,zw].
    roi_center_3d is expected in input image coordinates [B,3]=(z,y,x).
    Invalid roi falls back to center crop.
    """
    if feat.dim() != 5:
        raise ValueError(f"_roi_crop_3d expects [B,C,D,H,W], got {tuple(feat.shape)}")
    bsz, _, d, h, w = feat.shape
    zd, zh, zw = [int(v) for v in size_zyx]
    if zd <= 0 or zh <= 0 or zw <= 0:
        return feat

    if roi_center_3d is None:
        roi_center_3d = torch.full((bsz, 3), -1.0, device=feat.device, dtype=torch.float32)
    else:
        roi_center_3d = torch.as_tensor(roi_center_3d, device=feat.device, dtype=torch.float32)
        if roi_center_3d.ndim == 1:
            roi_center_3d = roi_center_3d.unsqueeze(0)
        if roi_center_3d.shape[0] == 1 and bsz > 1:
            roi_center_3d = roi_center_3d.expand(bsz, -1)
        if roi_center_3d.shape[0] != bsz or roi_center_3d.shape[1] != 3:
            roi_center_3d = torch.full((bsz, 3), -1.0, device=feat.device, dtype=torch.float32)

    img_d, img_h, img_w = image_dhw if image_dhw is not None else (d, h, w)
    out_list = []
    for i in range(bsz):
        rc = roi_center_3d[i]
        if _valid_roi_3d(rc):
            cz = float(rc[0].item()) * (float(d) / max(float(img_d), 1.0))
            cy = float(rc[1].item()) * (float(h) / max(float(img_h), 1.0))
            cx = float(rc[2].item()) * (float(w) / max(float(img_w), 1.0))
        else:
            cz = (d - 1) / 2.0
            cy = (h - 1) / 2.0
            cx = (w - 1) / 2.0
        cz, cy, cx = int(round(cz)), int(round(cy)), int(round(cx))

        z0, y0, x0 = cz - zd // 2, cy - zh // 2, cx - zw // 2
        z1, y1, x1 = z0 + zd, y0 + zh, x0 + zw

        sz0, sy0, sx0 = max(0, z0), max(0, y0), max(0, x0)
        sz1, sy1, sx1 = min(d, z1), min(h, y1), min(w, x1)
        crop = feat[i : i + 1, :, sz0:sz1, sy0:sy1, sx0:sx1]

        pz0, py0, px0 = max(0, -z0), max(0, -y0), max(0, -x0)
        pz1, py1, px1 = max(0, z1 - d), max(0, y1 - h), max(0, x1 - w)
        if pz0 or pz1 or py0 or py1 or px0 or px1:
            crop = F.pad(crop, (px0, px1, py0, py1, pz0, pz1), mode="replicate")
        if crop.shape[-3:] != (zd, zh, zw):
            crop = F.interpolate(crop, size=(zd, zh, zw), mode="trilinear", align_corners=False)
        out_list.append(crop)
    return torch.cat(out_list, dim=0)


class MedicalVLM(nn.Module):
    """
    Vision branch only.
    Output visual tokens [B, L, D] for later concatenation with LLM text embeddings.
    """

    def __init__(
        self,
        encoder_checkpoint: Optional[str] = None,
        encoder_output_stage: int = 4,
        encoder_target_spatial: int = 28,
        use_pooling: bool = True,
        pool_size: int = 12,
        global_pool_size: int = 8,
        local_crop_size: int = 10,
        global_pool_size_3d: tuple[int, int, int] = (2, 4, 4),
        local_crop_size_3d: tuple[int, int, int] = (8, 32, 32),
        ablation_mode: str = "full",
        bridge_d_model: int = 2560,
        bridge_bidirectional: bool = True,
        roi_side: Optional[int] = None,
        use_cmi: bool = False,
        cmi_compress_to: Optional[int] = None,
        cmi_d_state: int = 64,
        spatial_dims: int = 2,
        vision_bridge_type: str = "vim",
    ):
        super().__init__()
        self.spatial_dims = int(spatial_dims)
        if self.spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {self.spatial_dims}")

        if self.spatial_dims == 3:
            self.encoder = build_nnunet_encoder_3d(
                output_stage_index=encoder_output_stage,
                in_channels=1,
            )
        else:
            self.encoder = build_nnunet_encoder_light(
                checkpoint_path=encoder_checkpoint,
                output_stage_index=encoder_output_stage,
                target_spatial_size=encoder_target_spatial,
            )

        # Backward compatibility.
        if local_crop_size is None:
            local_crop_size = int(roi_side) if roi_side is not None else 10

        self.global_pool_size = max(1, int(global_pool_size))
        self.local_crop_size = max(1, int(local_crop_size))
        self.global_pool_size_3d = tuple(int(v) for v in global_pool_size_3d)
        self.local_crop_size_3d = tuple(int(v) for v in local_crop_size_3d)

        self.ablation_mode = str(ablation_mode).strip().lower()
        if self.ablation_mode not in {"full", "global_only", "local_only"}:
            raise ValueError(
                f"ablation_mode must be one of ['full', 'global_only', 'local_only'], got: {ablation_mode}"
            )

        if self.spatial_dims == 3:
            self.global_pool_3d = nn.AdaptiveAvgPool3d(self.global_pool_size_3d)
            self.global_pool = None
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((self.global_pool_size, self.global_pool_size))
            self.global_pool_3d = None

        bridge_kind = str(vision_bridge_type).strip().lower()
        self.vision_bridge_type = bridge_kind
        if bridge_kind == "transformer":
            self.bridge = TransformerBridge(
                in_channels=self.encoder.out_channels,
                d_model=bridge_d_model,
                bidirectional=bridge_bidirectional,
            )
        else:
            self.bridge = VimBridge(
                in_channels=self.encoder.out_channels,
                d_model=bridge_d_model,
                bidirectional=bridge_bidirectional,
            )
        self.bridge_d_model = bridge_d_model

        if self.spatial_dims == 3:
            g = int(self.global_pool_size_3d[0] * self.global_pool_size_3d[1] * self.global_pool_size_3d[2])
            l = int(self.local_crop_size_3d[0] * self.local_crop_size_3d[1] * self.local_crop_size_3d[2])
        else:
            g = self.global_pool_size * self.global_pool_size
            l = self.local_crop_size * self.local_crop_size
        if self.ablation_mode == "global_only":
            self.visual_seq_len = g
        elif self.ablation_mode == "local_only":
            self.visual_seq_len = l
        else:
            self.visual_seq_len = g + l

        self.cmi_connector: Optional[CMIConnector] = None
        if use_cmi:
            self.cmi_connector = CMIConnector(
                d_visual=bridge_d_model,
                d_text=bridge_d_model,
                d_model=bridge_d_model,
                d_state=cmi_d_state,
                compress_to=cmi_compress_to,
            )

    def forward(
        self,
        image: torch.Tensor,
        roi_center: Optional[torch.Tensor] = None,
        roi_center_3d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        2D:
          image: [B,1,H,W]
          roi_center: [B,2] in input image coords (y,x)
        3D:
          image: [B,1,D,H,W]
          roi_center_3d: [B,3] in input image coords (z,y,x)
        return:
          [B, L, D]
        """
        feat = self.encoder(image)
        seq_parts = []

        if self.spatial_dims == 3:
            if image.dim() != 5:
                raise ValueError(f"3D mode expects image [B,1,D,H,W], got {tuple(image.shape)}")
            image_dhw = (int(image.shape[-3]), int(image.shape[-2]), int(image.shape[-1]))
            if self.ablation_mode in {"full", "global_only"}:
                feat_global = self.global_pool_3d(feat)
                seq_global = feat_global.flatten(2).transpose(1, 2)  # [B, N_global, C]
                seq_parts.append(seq_global)
            if self.ablation_mode in {"full", "local_only"}:
                feat_local = _roi_crop_3d(
                    feat,
                    self.local_crop_size_3d,
                    roi_center_3d=roi_center_3d,
                    image_dhw=image_dhw,
                )
                seq_local = feat_local.flatten(2).transpose(1, 2)  # [B, N_local, C]
                seq_parts.append(seq_local)
        else:
            if image.dim() != 4:
                raise ValueError(f"2D mode expects image [B,1,H,W], got {tuple(image.shape)}")
            image_hw = (int(image.shape[-2]), int(image.shape[-1]))
            if self.ablation_mode in {"full", "global_only"}:
                feat_global = self.global_pool(feat)
                seq_global = feat_global.flatten(2).transpose(1, 2)  # [B, G*G, C]
                seq_parts.append(seq_global)
            if self.ablation_mode in {"full", "local_only"}:
                feat_local = _roi_crop_2d(
                    feat,
                    self.local_crop_size,
                    roi_center=roi_center,
                    image_hw=image_hw,
                )
                seq_local = feat_local.flatten(2).transpose(1, 2)  # [B, S*S, C]
                seq_parts.append(seq_local)

        if not seq_parts:
            raise RuntimeError(f"No visual stream enabled under ablation_mode={self.ablation_mode}")

        seq_combined = seq_parts[0] if len(seq_parts) == 1 else torch.cat(seq_parts, dim=1)
        return self.bridge(seq_combined)


def build_medical_vlm_from_config(config: dict) -> MedicalVLM:
    """Build MedicalVLM from config (e.g., config/paths.yaml)."""
    # Paper-consistent default: 8x8 global stream in 2D.
    global_pool_size = config.get("global_pool_size", 8)
    if global_pool_size is None:
        global_pool_size = 8

    local_crop_size = config.get("local_crop_size", None)
    if local_crop_size is None:
        local_crop_size = config.get("roi_side", None)
    if local_crop_size is None:
        local_crop_size = 10

    global_pool_size_3d = _as_tuple3(config.get("global_pool_size_3d", (2, 4, 4)), (2, 4, 4))
    local_crop_size_3d = _as_tuple3(config.get("local_crop_size_3d", (8, 32, 32)), (8, 32, 32))
    ablation_mode = str(config.get("ablation_mode", "full")).strip().lower()
    spatial_dims = int(config.get("spatial_dims", 2))
    vision_bridge_type = str(config.get("vision_bridge_type", "vim")).strip().lower()

    return MedicalVLM(
        encoder_checkpoint=config.get("nnunet_encoder_checkpoint"),
        encoder_output_stage=config.get("encoder_output_stage", 4),
        encoder_target_spatial=config.get("encoder_target_spatial", 28),
        use_pooling=config.get("use_pooling", True),
        pool_size=config.get("pool_size", 12),
        global_pool_size=int(global_pool_size),
        local_crop_size=int(local_crop_size),
        global_pool_size_3d=global_pool_size_3d,
        local_crop_size_3d=local_crop_size_3d,
        ablation_mode=ablation_mode,
        bridge_d_model=config.get("bridge_d_model", 2560),
        bridge_bidirectional=config.get("bridge_bidirectional", True),
        roi_side=config.get("roi_side"),
        use_cmi=config.get("use_cmi", False),
        cmi_compress_to=config.get("cmi_compress_to"),
        cmi_d_state=config.get("cmi_d_state", 64),
        spatial_dims=spatial_dims,
        vision_bridge_type=vision_bridge_type,
    )
