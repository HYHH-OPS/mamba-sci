"""
Medical VLM visual forward:
nnU-Net encoder -> Global-Local dual-stream tokenization -> VimBridge -> optional CMI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Project modules (run from repo root or include repo in PYTHONPATH)
try:
    from vision import build_nnunet_encoder_light
    from bridge import VimBridge, CMIConnector
except ImportError:
    import sys

    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from vision import build_nnunet_encoder_light
    from bridge import VimBridge, CMIConnector


def _center_crop_2d(feat: torch.Tensor, size: int) -> torch.Tensor:
    """Center crop [B, C, H, W] to [B, C, size, size]."""
    _, _, h, w = feat.shape
    if h < size or w < size:
        return feat
    start_h = (h - size) // 2
    start_w = (w - size) // 2
    return feat[:, :, start_h : start_h + size, start_w : start_w + size]


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
        bridge_d_model: int = 2560,
        bridge_bidirectional: bool = True,
        roi_side: Optional[int] = None,
        use_cmi: bool = False,
        cmi_compress_to: Optional[int] = None,
        cmi_d_state: int = 64,
    ):
        super().__init__()
        self.encoder = build_nnunet_encoder_light(
            checkpoint_path=encoder_checkpoint,
            output_stage_index=encoder_output_stage,
            target_spatial_size=encoder_target_spatial,
        )

        # Backward compatibility with old single-stream config keys.
        if global_pool_size is None:
            global_pool_size = int(pool_size) if use_pooling else 8
        if local_crop_size is None:
            local_crop_size = int(roi_side) if roi_side is not None else 10

        self.global_pool_size = max(1, int(global_pool_size))
        self.local_crop_size = max(1, int(local_crop_size))

        self.global_pool = nn.AdaptiveAvgPool2d((self.global_pool_size, self.global_pool_size))

        self.bridge = VimBridge(
            in_channels=self.encoder.out_channels,
            d_model=bridge_d_model,
            bidirectional=bridge_bidirectional,
        )
        self.bridge_d_model = bridge_d_model

        # Global tokens + Local tokens
        self.visual_seq_len = (
            self.global_pool_size * self.global_pool_size
            + self.local_crop_size * self.local_crop_size
        )

        self.cmi_connector: Optional[CMIConnector] = None
        if use_cmi:
            self.cmi_connector = CMIConnector(
                d_visual=bridge_d_model,
                d_text=bridge_d_model,
                d_model=bridge_d_model,
                d_state=cmi_d_state,
                compress_to=cmi_compress_to,
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: [B, 1, H, W]
        return: [B, L, D], where L = G*G + S*S
        """
        feat = self.encoder(image)  # [B, C, Hf, Wf]

        # 1) Global stream: coarse context (forest)
        feat_global = self.global_pool(feat)
        seq_global = feat_global.flatten(2).transpose(1, 2)  # [B, G*G, C]

        # 2) Local stream: center crop high-resolution details (trees)
        feat_local = _center_crop_2d(feat, self.local_crop_size)
        seq_local = feat_local.flatten(2).transpose(1, 2)  # [B, S*S, C]

        # 3) Fuse tokens and feed bridge
        seq_combined = torch.cat([seq_global, seq_local], dim=1)  # [B, G*G+S*S, C]
        return self.bridge(seq_combined)


def build_medical_vlm_from_config(config: dict) -> MedicalVLM:
    """Build MedicalVLM from config (e.g., config/paths.yaml)."""
    global_pool_size = config.get("global_pool_size", None)
    if global_pool_size is None:
        global_pool_size = config.get("pool_size", 12)

    local_crop_size = config.get("local_crop_size", None)
    if local_crop_size is None:
        local_crop_size = config.get("roi_side", 10)

    return MedicalVLM(
        encoder_checkpoint=config.get("nnunet_encoder_checkpoint"),
        encoder_output_stage=config.get("encoder_output_stage", 4),
        encoder_target_spatial=config.get("encoder_target_spatial", 28),
        use_pooling=config.get("use_pooling", True),
        pool_size=config.get("pool_size", 12),
        global_pool_size=global_pool_size,
        local_crop_size=local_crop_size,
        bridge_d_model=config.get("bridge_d_model", 2560),
        bridge_bidirectional=config.get("bridge_bidirectional", True),
        roi_side=config.get("roi_side"),
        use_cmi=config.get("use_cmi", False),
        cmi_compress_to=config.get("cmi_compress_to"),
        cmi_d_state=config.get("cmi_d_state", 64),
    )
