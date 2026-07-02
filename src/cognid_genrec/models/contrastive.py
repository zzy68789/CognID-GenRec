from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def info_nce_loss(
    user_embeddings: torch.Tensor,
    positive_item_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    logits = user_embeddings @ positive_item_embeddings.T
    logits = logits / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


class ItemEmbeddingTable(nn.Module):
    def __init__(self, num_items: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(item_ids)
