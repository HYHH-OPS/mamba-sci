"""
nnU-Net 编码器：轻下采样，保留 28×28 / 32×32 等高分辨率特征。

与 ARCHITECTURE.md 一致：不把特征压到 14×14，利用 Mamba 线性复杂度保留更大空间尺寸。

- 2D 默认 8 stages 时：stage 0→512, 1→256, 2→128, 3→64, 4→32, 5→16, 6→8, 7→4
- 取 stage 4 输出 → 32×32；可选 interpolate 到 28×28
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


# 可选：使用 nnUNet 的 dynamic_network_architectures
try:
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    from dynamic_network_architectures.building_blocks.helper import get_matching_conv
    _HAS_DNA = True
except ImportError:
    _HAS_DNA = False
    PlainConvUNet = None


# 2D 默认 plans 与 Dataset503 一致（8 stages, 512 patch）
DEFAULT_2D_FEATURES = [32, 64, 128, 256, 512, 512, 512, 512]
DEFAULT_2D_STRIDES = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
DEFAULT_3D_FEATURES = [24, 48, 96, 192, 256]
DEFAULT_3D_STRIDES = [(1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)]


def _get_stage_spatial_size(input_size: int, strides_per_stage: list[list[int]]) -> list[int]:
    """返回每个 stage 输出的空间边长（2D 假设 H=W）。"""
    sizes = [input_size]
    for s in strides_per_stage:
        stride = s[0] if isinstance(s[0], int) else s[0][0]
        sizes.append(sizes[-1] // stride)
    return sizes


class NNUnetEncoderLight(nn.Module):
    """
    nnU-Net 编码器，在指定 stage 截断，输出高分辨率特征图（如 32×32 或 28×28）。

    - output_stage_index: 取到第几层 encoder stage（0-based）。4 → 32×32（512/16）
    - target_spatial_size: 若设（如 28），则对输出做 interpolate 到 28×28
    - 若不使用 nnUNet 权重，则随机初始化（仍可用于结构验证）。
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_stages: int = 8,
        features_per_stage: list[int] | None = None,
        strides_per_stage: list[list[int]] | None = None,
        output_stage_index: int = 4,
        target_spatial_size: Optional[int] = 28,
        kernel_sizes: list[list[int]] | None = None,
        use_nnunet_conv: bool = True,
    ):
        super().__init__()
        self.output_stage_index = output_stage_index
        self.target_spatial_size = target_spatial_size
        features = features_per_stage or DEFAULT_2D_FEATURES
        strides = strides_per_stage or DEFAULT_2D_STRIDES
        kernel_sizes = kernel_sizes or [[3, 3]] * num_stages

        self.stages = nn.ModuleList()
        ch_in = in_channels
        for i in range(min(num_stages, output_stage_index + 1)):
            ch_out = features[i]
            s = strides[i]
            k = kernel_sizes[i] if isinstance(kernel_sizes[i][0], int) else kernel_sizes[i]
            stride = s[0] if isinstance(s[0], int) else s[0]
            pad = k[0] // 2
            block = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.InstanceNorm2d(ch_out, eps=1e-5, affine=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(ch_out, ch_out, kernel_size=k, stride=1, padding=pad, bias=True),
                nn.InstanceNorm2d(ch_out, eps=1e-5, affine=True),
                nn.LeakyReLU(inplace=True),
            )
            self.stages.append(block)
            ch_in = ch_out
        self.out_channels = ch_in
        self._spatial_sizes = _get_stage_spatial_size(512, strides)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]，如 [B, 1, 512, 512]
        for stage in self.stages:
            x = stage(x)
        if self.target_spatial_size is not None and x.shape[2] != self.target_spatial_size:
            x = nn.functional.interpolate(
                x,
                size=(self.target_spatial_size, self.target_spatial_size),
                mode="bilinear",
                align_corners=False,
            )
        return x

    @property
    def output_spatial_size(self) -> int:
        if self.target_spatial_size is not None:
            return self.target_spatial_size
        return self._spatial_sizes[self.output_stage_index + 1]


class NNUnetEncoder3D(nn.Module):
    """
    Lightweight 3D nnU-Net-like encoder.
    Input:  [B, 1, D, H, W]
    Output: [B, C, Df, Hf, Wf]
    """

    def __init__(
        self,
        in_channels: int = 1,
        features_per_stage: list[int] | None = None,
        strides_per_stage: list[tuple[int, int, int]] | None = None,
        output_stage_index: int = 4,
    ):
        super().__init__()
        features = features_per_stage or DEFAULT_3D_FEATURES
        strides = strides_per_stage or DEFAULT_3D_STRIDES
        max_stage = min(output_stage_index + 1, len(features), len(strides))

        self.stages = nn.ModuleList()
        ch_in = in_channels
        for i in range(max_stage):
            ch_out = int(features[i])
            s = tuple(int(v) for v in strides[i])
            block = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=s, padding=1, bias=True),
                nn.InstanceNorm3d(ch_out, eps=1e-5, affine=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.InstanceNorm3d(ch_out, eps=1e-5, affine=True),
                nn.LeakyReLU(inplace=True),
            )
            self.stages.append(block)
            ch_in = ch_out
        self.out_channels = ch_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


def build_nnunet_encoder_light(
    checkpoint_path: Optional[str | Path] = None,
    output_stage_index: int = 4,
    target_spatial_size: int = 28,
    in_channels: int = 1,
    strict: bool = False,
) -> NNUnetEncoderLight:
    """
    构建轻下采样 nnU-Net 编码器；若提供 nnUNet 2D checkpoint，则尽量加载匹配的 encoder 权重。

    - checkpoint_path: nnUNet 2D checkpoint（如 fold_0/checkpoint_best.pth）
    - output_stage_index: 4 → 32×32 再可 interpolate 到 28×28
    - strict: 加载时是否严格匹配 key（encoder 可能只匹配前几层）
    """
    model = NNUnetEncoderLight(
        in_channels=in_channels,
        output_stage_index=output_stage_index,
        target_spatial_size=target_spatial_size,
    )
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.exists():
            # nnUNet 的 pth 含 numpy 等对象，需 weights_only=False（仅加载自信任的本地权重）
            state = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "network_weights" in state:
                state = state["network_weights"]
            # 只取 encoder 部分中与我们 stages 对应的 key（前缀如 stages.0, stages.1, ...）
            our_state = model.state_dict()
            loaded = {}
            for k, v in state.items():
                if k.startswith("encoder.stages."):
                    parts = k.split(".")
                    try:
                        idx = int(parts[2])
                        if idx <= output_stage_index:
                            new_k = "stages." + ".".join(parts[2:])
                            if new_k in our_state and our_state[new_k].shape == v.shape:
                                loaded[new_k] = v
                    except (ValueError, IndexError):
                        pass
                elif k.startswith("stages."):
                    try:
                        idx = int(k.split(".")[1])
                        if idx <= output_stage_index and k in our_state and our_state[k].shape == v.shape:
                            loaded[k] = v
                    except (ValueError, IndexError):
                        pass
            if loaded:
                model.load_state_dict(loaded, strict=strict)
    return model


def build_nnunet_encoder_3d(
    output_stage_index: int = 4,
    in_channels: int = 1,
) -> NNUnetEncoder3D:
    """
    Build 3D encoder for true 3D patches.
    """
    return NNUnetEncoder3D(
        in_channels=in_channels,
        output_stage_index=output_stage_index,
    )
