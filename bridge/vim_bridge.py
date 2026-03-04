"""
Vision Mamba bridge: sequence projection + (bi-)Mamba modeling + learnable queries.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Optional: mamba_ssm backend
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    _HAS_MAMBA_SSM = True
except ImportError:
    _HAS_MAMBA_SSM = False
    Mamba = None


class VimBlock(nn.Module):
    """Single directional Mamba block with residual + LayerNorm."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
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
    Bridge that consumes 1D visual sequence tokens [B, L, C]
    and outputs projected/modelled visual tokens [B, L, d_model].
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
        self.norm = nn.LayerNorm(d_model)
        self.bidirectional = bidirectional

        self.vim_fwd = VimBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        if bidirectional:
            self.vim_bwd = VimBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.out_proj = nn.Linear(d_model * 2, d_model)
        else:
            self.vim_bwd = None
            self.out_proj = nn.Identity()

        # Learnable clinical queries (Q-Former style).
        self.num_queries = 32
        self.query_tokens = nn.Parameter(torch.randn(self.num_queries, d_model))
        self.latest_queries_out: torch.Tensor | None = None

    def forward(self, x_1d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_1d: [B, L, C]
        Returns:
            tokens: [B, L, d_model]
        """
        if x_1d.dim() != 3:
            raise ValueError(f"VimBridge expects [B, L, C], got shape={tuple(x_1d.shape)}")

        bsz, seq_len, c_in = x_1d.shape
        if c_in != self.proj.in_features:
            raise ValueError(
                f"VimBridge input channel mismatch: got C={c_in}, expected {self.proj.in_features}"
            )

        x = self.norm(self.proj(x_1d))  # [B, L, D]

        # Append learnable queries after image tokens.
        queries = self.query_tokens.unsqueeze(0).expand(bsz, -1, -1)  # [B, Q, D]
        seq = torch.cat([x, queries], dim=1)  # [B, L+Q, D]

        if self.bidirectional:
            seq_fwd = self.vim_fwd(seq)
            seq_bwd = self.vim_bwd(torch.flip(seq, dims=[1]))
            seq_bwd = torch.flip(seq_bwd, dims=[1])
            seq = self.out_proj(torch.cat([seq_fwd, seq_bwd], dim=-1))
        else:
            seq = self.vim_fwd(seq)

        tokens = seq[:, :-self.num_queries, :]      # [B, L, D]
        queries_out = seq[:, -self.num_queries:, :]  # [B, Q, D]
        self.latest_queries_out = queries_out
        return tokens
