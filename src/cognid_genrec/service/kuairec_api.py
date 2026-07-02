from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from cognid_genrec.retrieval.ann_index import NumpyANNIndex


class KuaiRecRecommendRequest(BaseModel):
    user_id: str
    history_item_ids: list[str] = Field(default_factory=list)
    history_actions: list[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)


class KuaiRecRecommendedItem(BaseModel):
    item_id: str
    score: float
    semantic_id: str
    source: str
    action_context: str
    ann_rank: int
    reason: str


class KuaiRecRecommendResponse(BaseModel):
    user_id: str
    items: list[KuaiRecRecommendedItem]


class KuaiRecRecommendationEngine:
    def __init__(
        self,
        index: NumpyANNIndex,
        item_embeddings: dict[str, np.ndarray],
        semantic_by_item: dict[str, str],
        topic_by_item: dict[str, str],
    ) -> None:
        self.index = index
        self.item_embeddings = item_embeddings
        self.semantic_by_item = semantic_by_item
        self.topic_by_item = topic_by_item

    @classmethod
    def load(cls, data_dir: str | Path) -> "KuaiRecRecommendationEngine":
        root = Path(data_dir)
        if (root / "ann_items.npy").exists() and (root / "ann_item_ids.json").exists():
            embeddings = np.load(root / "ann_items.npy")
            item_ids = json.loads((root / "ann_item_ids.json").read_text(encoding="utf-8"))
            index = NumpyANNIndex(item_ids=item_ids, embeddings=embeddings)
            item_embeddings = {
                str(item_id): embeddings[row_index].astype(np.float32)
                for row_index, item_id in enumerate(item_ids)
            }
            semantic_by_item = _load_semantic_mapping(root)
            topic_by_item = _load_topics(root)
            return cls(index, item_embeddings, semantic_by_item, topic_by_item)
        return cls.sample()

    @classmethod
    def sample(cls) -> "KuaiRecRecommendationEngine":
        item_ids = ["10", "20", "30", "40"]
        embeddings = np.array(
            [
                [1.0, 0.0, 0.1, 0.0],
                [0.7, 0.2, 0.1, 0.1],
                [0.0, 1.0, 0.2, 0.0],
                [0.1, 0.2, 1.0, 0.3],
            ],
            dtype=np.float32,
        )
        return cls(
            index=NumpyANNIndex(item_ids=item_ids, embeddings=embeddings),
            item_embeddings={item_id: embeddings[index] for index, item_id in enumerate(item_ids)},
            semantic_by_item={
                "10": "0-0-1",
                "20": "0-1-1",
                "30": "1-0-1",
                "40": "2-0-0",
            },
            topic_by_item={
                "10": "food",
                "20": "food",
                "30": "music",
                "40": "sports",
            },
        )

    def recommend(self, request: KuaiRecRecommendRequest) -> KuaiRecRecommendResponse:
        query = self._query_embedding(request.history_item_ids)
        raw_results = self.index.search(query, top_k=request.top_k + len(request.history_item_ids) + 5)
        history = {str(item_id) for item_id in request.history_item_ids}
        action_context = request.history_actions[-1] if request.history_actions else "unknown"
        items = []
        for item_id, score in raw_results:
            if item_id in history:
                continue
            items.append(
                KuaiRecRecommendedItem(
                    item_id=item_id,
                    score=round(float(score), 6),
                    semantic_id=self.semantic_by_item.get(item_id, ""),
                    source="ann_transformer",
                    action_context=action_context,
                    ann_rank=len(items) + 1,
                    reason=self._reason(item_id, action_context),
                )
            )
            if len(items) >= request.top_k:
                break
        return KuaiRecRecommendResponse(user_id=request.user_id, items=items)

    def _query_embedding(self, history_item_ids: list[str]) -> np.ndarray:
        vectors = [
            self.item_embeddings[str(item_id)]
            for item_id in history_item_ids
            if str(item_id) in self.item_embeddings
        ]
        if vectors:
            return np.mean(np.stack(vectors), axis=0).astype(np.float32)
        return np.mean(np.stack(list(self.item_embeddings.values())), axis=0).astype(np.float32)

    def _reason(self, item_id: str, action_context: str) -> str:
        topic = self.topic_by_item.get(item_id, "unknown")
        return (
            f"ANN retrieved short-video candidate in topic={topic}; "
            f"recent action context={action_context}."
        )


def create_kuairec_app(data_dir: str | Path = "data/processed/kuairec") -> FastAPI:
    engine = KuaiRecRecommendationEngine.load(data_dir)
    app = FastAPI(title="CognID-GenRec")

    @app.post("/recommend", response_model=KuaiRecRecommendResponse)
    def recommend(request: KuaiRecRecommendRequest) -> KuaiRecRecommendResponse:
        return engine.recommend(request)

    return app


def _load_semantic_mapping(root: Path) -> dict[str, str]:
    path = root / "semantic_ids.parquet"
    if not path.exists():
        return {}
    mapping = pd.read_parquet(path)
    if "semantic_id_str" not in mapping.columns:
        mapping["semantic_id_str"] = mapping["semantic_id"].map(str)
    return dict(zip(mapping["item_id"].astype(str), mapping["semantic_id_str"].astype(str), strict=False))


def _load_topics(root: Path) -> dict[str, str]:
    for filename in ["items.parquet", "item_features.parquet"]:
        path = root / filename
        if path.exists():
            items = pd.read_parquet(path)
            if {"item_id", "topic"} <= set(items.columns):
                return dict(zip(items["item_id"].astype(str), items["topic"].astype(str), strict=False))
    return {}
