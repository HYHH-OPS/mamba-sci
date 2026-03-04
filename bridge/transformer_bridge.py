"""
Transformer bridge baseline:
sequence projection + TransformerEncoder + learnable queries.

Interface is intentionally aligned with VimBridge for fair ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBridge(nn.Module):
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
        num_layers: int | None = None,
        nhead: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        del d_state, d_conv, expand  # kept in signature for parity with VimBridge

        self.proj = nn.Linear(in_channels, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.bidirectional = bidirectional

        if d_model % nhead != 0:
            # Fallback to a valid head count.
            nhead = 1
        if num_layers is None:
            num_layers = 2 if bidirectional else 1

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Learnable clinical queries (Q-Former style) - same as VimBridge.
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
            raise ValueError(f"TransformerBridge expects [B, L, C], got shape={tuple(x_1d.shape)}")

        bsz, _, c_in = x_1d.shape
        if c_in != self.proj.in_features:
            raise ValueError(
                f"TransformerBridge input channel mismatch: got C={c_in}, expected {self.proj.in_features}"
            )

        x = self.norm(self.proj(x_1d))  # [B, L, D]

        # Append learnable queries after image tokens.
        queries = self.query_tokens.unsqueeze(0).expand(bsz, -1, -1)  # [B, Q, D]
        seq = torch.cat([x, queries], dim=1)  # [B, L+Q, D]
        seq = self.encoder(seq)

        tokens = seq[:, :-self.num_queries, :]      # [B, L, D]
        queries_out = seq[:, -self.num_queries :, :]  # [B, Q, D]
        self.latest_queries_out = queries_out
        return tokens

