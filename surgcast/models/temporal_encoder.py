from __future__ import annotations

import torch
import torch.nn as nn


class CausalTemporalTransformer(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, layers: int = 6, heads: int = 8, dropout: float = 0.1, max_seq_len: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dropout=dropout,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.input_proj(x) + self.pos[:, : x.size(1)])
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        return self.encoder(x, mask=mask)
