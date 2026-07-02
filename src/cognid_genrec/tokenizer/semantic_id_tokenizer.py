from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.nn import functional as F

from cognid_genrec.tokenizer.item_text_encoder import TfidfItemTextEncoder


class SemanticIDTokenizer:
    def __init__(
        self,
        num_levels: int = 3,
        clusters_per_level: Sequence[int] = (8, 16, 32),
        random_state: int = 42,
    ) -> None:
        if num_levels < 1:
            raise ValueError("num_levels must be positive")
        if not clusters_per_level:
            raise ValueError("clusters_per_level must not be empty")
        self.num_levels = num_levels
        self.clusters_per_level = tuple(clusters_per_level)
        self.random_state = random_state
        self.encoder = TfidfItemTextEncoder()

    def fit_transform(self, items: pd.DataFrame) -> pd.DataFrame:
        if items.empty:
            raise ValueError("items must not be empty")
        features = self.encoder.fit_transform(items)

        semantic_ids: list[list[int]] = [[] for _ in range(len(items))]
        for level in range(self.num_levels):
            requested_clusters = self.clusters_per_level[
                min(level, len(self.clusters_per_level) - 1)
            ]
            n_clusters = max(1, min(requested_clusters, len(items)))
            labels = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state + level,
                n_init=10,
            ).fit_predict(features)
            for row_index, label in enumerate(labels):
                semantic_ids[row_index].append(int(label))

        mapping = items[["item_id", "topic"]].copy()
        mapping["semantic_id"] = semantic_ids
        mapping["semantic_id_str"] = ["-".join(map(str, values)) for values in semantic_ids]
        return mapping


class _ResidualQuantizer(nn.Module):
    def __init__(
        self,
        num_levels: int,
        codebook_size: int,
        embed_dim: int,
        commitment_weight: float,
        random_state: int,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.commitment_weight = commitment_weight
        self.random_state = random_state
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.empty(codebook_size, embed_dim)) for _ in range(num_levels)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        for codebook in self.codebooks:
            nn.init.uniform_(codebook, -0.05, 0.05, generator=generator)

    @torch.no_grad()
    def initialize_from_kmeans(self, embeddings: torch.Tensor) -> None:
        residual = embeddings.detach().cpu().numpy()
        for level, codebook in enumerate(self.codebooks):
            distinct_rows = np.unique(np.round(residual, decimals=8), axis=0).shape[0]
            n_clusters = max(1, min(self.codebook_size, len(residual), distinct_rows))
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state + level,
                n_init=10,
            ).fit(residual)
            labels = kmeans.labels_
            centroids = np.zeros((self.codebook_size, self.embed_dim), dtype=np.float32)
            cluster_centers = kmeans.cluster_centers_.astype(np.float32)
            centroids[:n_clusters] = cluster_centers
            if n_clusters < self.codebook_size:
                fill_count = self.codebook_size - n_clusters
                fill_indices = np.arange(fill_count) % n_clusters
                centroids[n_clusters:] = cluster_centers[fill_indices]
            codebook.copy_(torch.from_numpy(centroids).to(codebook.device))
            residual = residual - cluster_centers[labels]

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = embeddings
        quantized_parts: list[torch.Tensor] = []
        semantic_ids: list[torch.Tensor] = []
        quantize_loss = embeddings.new_tensor(0.0)

        for codebook in self.codebooks:
            distances = torch.cdist(residual, codebook)
            ids = distances.argmin(dim=1)
            quantized = codebook[ids]
            quantized_parts.append(quantized)
            semantic_ids.append(ids)
            quantize_loss = quantize_loss + F.mse_loss(quantized, residual.detach())
            quantize_loss = quantize_loss + self.commitment_weight * F.mse_loss(
                residual, quantized.detach()
            )
            residual = residual - quantized

        quantized_sum = torch.stack(quantized_parts, dim=0).sum(dim=0)
        quantized_ste = embeddings + (quantized_sum - embeddings).detach()
        return quantized_ste, torch.stack(semantic_ids, dim=1), quantize_loss


class _TinyRQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dim: int,
        num_levels: int,
        codebook_size: int,
        commitment_weight: float,
        random_state: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.quantizer = _ResidualQuantizer(
            num_levels=num_levels,
            codebook_size=codebook_size,
            embed_dim=embed_dim,
            commitment_weight=commitment_weight,
            random_state=random_state,
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(batch)
        quantized, semantic_ids, quantize_loss = self.quantizer(embeddings)
        reconstruction = self.decoder(quantized)
        return reconstruction, semantic_ids, quantize_loss

    @torch.no_grad()
    def semantic_ids(self, batch: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(batch)
        _, semantic_ids, _ = self.quantizer(embeddings)
        return semantic_ids


class RQVAESemanticIDTokenizer:
    def __init__(
        self,
        num_levels: int = 3,
        codebook_size: int = 32,
        embedding_dim: int = 16,
        hidden_dim: int | None = None,
        max_iter: int = 40,
        learning_rate: float = 0.01,
        commitment_weight: float = 0.25,
        random_state: int = 42,
    ) -> None:
        if num_levels < 1:
            raise ValueError("num_levels must be positive")
        if codebook_size < 1:
            raise ValueError("codebook_size must be positive")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.commitment_weight = commitment_weight
        self.random_state = random_state
        self.encoder = TfidfItemTextEncoder()

    def fit_transform(self, items: pd.DataFrame) -> pd.DataFrame:
        if items.empty:
            raise ValueError("items must not be empty")

        features = self.encoder.fit_transform(items).astype(np.float32).toarray()
        batch = torch.from_numpy(features)
        torch.manual_seed(self.random_state)
        hidden_dim = self.hidden_dim or max(self.embedding_dim * 2, min(128, features.shape[1]))
        model = _TinyRQVAE(
            input_dim=features.shape[1],
            embed_dim=self.embedding_dim,
            hidden_dim=hidden_dim,
            num_levels=self.num_levels,
            codebook_size=self.codebook_size,
            commitment_weight=self.commitment_weight,
            random_state=self.random_state,
        )

        with torch.no_grad():
            model.quantizer.initialize_from_kmeans(model.encoder(batch))

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        model.train()
        for _ in range(self.max_iter):
            reconstruction, _, quantize_loss = model(batch)
            reconstruction_loss = F.mse_loss(reconstruction, batch)
            loss = reconstruction_loss + quantize_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        semantic_ids_tensor = model.semantic_ids(batch)
        semantic_ids = [
            [int(value) for value in row]
            for row in semantic_ids_tensor.detach().cpu().tolist()
        ]

        mapping = items[["item_id", "topic"]].copy()
        mapping["semantic_id"] = semantic_ids
        mapping["semantic_id_str"] = ["-".join(map(str, values)) for values in semantic_ids]
        mapping["tokenizer_method"] = "rqvae"
        return mapping


def collision_rate(mapping: pd.DataFrame) -> float:
    if mapping.empty:
        return 0.0
    unique_ids = mapping["semantic_id_str"].nunique()
    return 1.0 - (unique_ids / len(mapping))


def topic_purity(mapping: pd.DataFrame) -> float:
    if mapping.empty:
        return 0.0
    dominant_topic_count = 0
    for _, group in mapping.groupby("semantic_id_str"):
        dominant_topic_count += int(group["topic"].value_counts().max())
    return dominant_topic_count / len(mapping)


def semantic_id_quality_metrics(mapping: pd.DataFrame) -> dict[str, float]:
    return {
        "item_count": float(len(mapping)),
        "semantic_id_unique": float(mapping["semantic_id_str"].nunique()),
        "collision_rate": collision_rate(mapping),
        "topic_purity": topic_purity(mapping),
    }


def write_semantic_id_quality_report(metrics: dict[str, float], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Semantic ID Quality",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for metric_name, value in metrics.items():
        lines.append(f"| {metric_name} | {float(value):.4f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
