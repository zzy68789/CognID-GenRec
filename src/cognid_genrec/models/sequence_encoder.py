from __future__ import annotations

import torch
from torch import nn


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_actions: int,
        hidden_dim: int = 128,
        max_len: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            num_heads = 1
        self.max_len = max_len
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.action_embedding = nn.Embedding(num_actions + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.time_projection = nn.Linear(1, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        item_ids: torch.Tensor,
        action_ids: torch.Tensor,
        time_deltas: torch.Tensor,
    ) -> torch.Tensor:
        if item_ids.shape != action_ids.shape or item_ids.shape != time_deltas.shape:
            raise ValueError("item_ids, action_ids, and time_deltas must have the same shape")
        batch_size, seq_len = item_ids.shape
        if seq_len > self.max_len:
            item_ids = item_ids[:, -self.max_len :]
            action_ids = action_ids[:, -self.max_len :]
            time_deltas = time_deltas[:, -self.max_len :]
            seq_len = self.max_len
        positions = torch.arange(seq_len, device=item_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        time_values = torch.log1p(time_deltas.float()).unsqueeze(-1)
        hidden = (
            self.item_embedding(item_ids)
            + self.action_embedding(action_ids)
            + self.position_embedding(positions)
            + self.time_projection(time_values)
        )
        hidden = self.layer_norm(hidden)
        padding_mask = item_ids.eq(0)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=item_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        lengths = item_ids.ne(0).sum(dim=1).clamp(min=1)
        batch_indices = torch.arange(batch_size, device=item_ids.device)
        return encoded[batch_indices, lengths - 1]
