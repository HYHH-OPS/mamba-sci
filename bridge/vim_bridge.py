"""
Vim (Vision Mamba) Bridge: 2D 特征图 → 1D 序列 → Mamba 建模 → 视觉 token 序列。

与 ARCHITECTURE.md 一致：将 nnU-Net 编码器输出的 [B, C, H, W] 展平为 [B, L, C]，
经线性投影到 d_model 后通过 Vision Mamba Block（双向或单向），输出 [B, L, D] 供 LLM 融合。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

# 可选：使用 mamba_ssm 的 Mamba 层
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    _HAS_MAMBA_SSM = True
except ImportError:
    _HAS_MAMBA_SSM = False
    Mamba = None


class VimBlock(nn.Module):
    """
    单向前向 Mamba 块；若 mamba_ssm 不可用则退化为 Linear + LayerNorm。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        if _HAS_MAMBA_SSM and Mamba is not None:
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mamba = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        return self.norm(x + self.mamba(x))


class VimBridge(nn.Module):
    """
    Vision Mamba 桥接器：2D feature map → 展平 → 投影 → 双向 Vim → 视觉 token 序列。

    输入: [B, C, H, W]（如 C=512, H=W=28）
    输出: [B, L, d_model]，L = H*W（如 28*28=784）
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        bidirectional: bool = True,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(d_model)
        self.vim_fwd = VimBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        if bidirectional:
            self.vim_bwd = VimBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.out_proj = nn.Linear(d_model * 2, d_model)
        else:
            self.vim_bwd = None
            self.out_proj = nn.Identity()
        # Learnable clinical queries（Q-Former 风格），用于侵润/风险分级等下游任务。
        # 注意：Mamba 为序列模型，queries 放在图像 token 之后，保证在处理 queries 前已“看完”整幅图像。
        self.num_queries = 32
        self.query_tokens = nn.Parameter(torch.randn(self.num_queries, d_model))
        self.latest_queries_out: torch.Tensor | None = None

    def forward(self, feat_2d: torch.Tensor) -> torch.Tensor:
        """
        feat_2d: [B, C, H, W]
        return: [B, H*W, d_model]（仅返回图像 token，queries_out 可通过 self.latest_queries_out 访问）
        """
        B, C, H, W = feat_2d.shape
        # 按行展平为 1D 序列（Vim 做法）
        x = feat_2d.flatten(2).permute(0, 2, 1)  # [B, L, C]
        x = self.norm(self.proj(x))  # [B, L, d_model]

        # 先图像 token，后 Query，符合 Mamba 序列依赖特性
        B, L, D = x.shape
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]
        seq = torch.cat([x, queries], dim=1)  # [B, L+Q, D]

        if self.bidirectional:
            seq_fwd = self.vim_fwd(seq)
            seq_bwd = self.vim_bwd(torch.flip(seq, dims=[1]))
            seq_bwd = torch.flip(seq_bwd, dims=[1])
            seq = self.out_proj(torch.cat([seq_fwd, seq_bwd], dim=-1))
        else:
            seq = self.vim_fwd(seq)

        # 拆分图像 token 与 Query 表示
        tokens = seq[:, :-self.num_queries, :]          # [B, L, D]
        queries_out = seq[:, -self.num_queries:, :]     # [B, Q, D]
        self.latest_queries_out = queries_out
        return tokens
