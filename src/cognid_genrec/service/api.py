from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from cognid_genrec.models.generative_retriever import GenerativeRetriever
from cognid_genrec.retrieval.diversity_reranker import DiversityReranker


class RecommendRequest(BaseModel):
    user_id: str
    history_item_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)


class RecommendedItem(BaseModel):
    item_id: str
    score: float
    semantic_id: list[int]
    reason: str
    source: str
    recall_path: list[str]
    rerank_reason: str


class RecommendResponse(BaseModel):
    user_id: str
    items: list[RecommendedItem]


class RecommendationEngine:
    def __init__(
        self,
        retriever: GenerativeRetriever,
        items: pd.DataFrame,
        semantic_mapping: pd.DataFrame,
    ) -> None:
        self.retriever = retriever
        self.items = items
        self.reranker = DiversityReranker()
        self.semantic_by_item = {
            str(row["item_id"]): row
            for row in semantic_mapping.to_dict("records")
        }

    @classmethod
    def load(cls, data_dir: str | Path) -> "RecommendationEngine":
        root = Path(data_dir)
        retriever = GenerativeRetriever.load(root / "generative_retriever.json")
        items = pd.read_csv(root / "items.csv")
        semantic_mapping = pd.read_parquet(root / "semantic_ids.parquet")
        return cls(retriever, items, semantic_mapping)

    def recommend(self, request: RecommendRequest) -> RecommendResponse:
        generated_semantic_ids = self.retriever.generate_semantic_ids(request.history_item_ids)
        raw_item_ids = self.retriever.recommend(
            history_item_ids=request.history_item_ids,
            top_k=request.top_k * 3,
        )
        candidates = [
            {"item_id": item_id, "score": 1.0 / (rank + 1)}
            for rank, item_id in enumerate(raw_item_ids)
        ]
        reranked = self.reranker.rerank(candidates, self.items, top_k=request.top_k)
        items = [
            RecommendedItem(
                item_id=candidate["item_id"],
                score=round(float(candidate["adjusted_score"]), 6),
                semantic_id=self._semantic_id(candidate["item_id"]),
                reason=self._reason(candidate),
                source="generative_rerank",
                recall_path=self._recall_path(
                    request.history_item_ids,
                    generated_semantic_ids,
                    candidate,
                ),
                rerank_reason=self._rerank_reason(candidate),
            )
            for candidate in reranked
        ]
        return RecommendResponse(user_id=request.user_id, items=items)

    def _semantic_id(self, item_id: str) -> list[int]:
        row = self.semantic_by_item[item_id]
        semantic_id = row.get("semantic_id")
        if hasattr(semantic_id, "tolist"):
            return [int(value) for value in semantic_id.tolist()]
        if isinstance(semantic_id, list):
            return [int(value) for value in semantic_id]
        return [int(value) for value in str(row["semantic_id_str"]).split("-")]

    def _semantic_id_str(self, item_id: str) -> str:
        return str(self.semantic_by_item[item_id]["semantic_id_str"])

    def _recall_path(
        self,
        history_item_ids: list[str],
        generated_semantic_ids: list[str],
        candidate: dict,
    ) -> list[str]:
        history_semantic_ids = [
            self.retriever.item_to_semantic[item_id]
            for item_id in history_item_ids
            if item_id in self.retriever.item_to_semantic
        ]
        return [
            "history_items=" + ",".join(history_item_ids),
            "history_semantic_ids=" + ",".join(history_semantic_ids),
            "generated_semantic_ids=" + ">".join(generated_semantic_ids),
            "decoded_item="
            + candidate["item_id"]
            + ";semantic_id="
            + self._semantic_id_str(candidate["item_id"]),
        ]

    def _reason(self, candidate: dict) -> str:
        if candidate.get("source_reason") == "long_tail_protection":
            return "Protected high-quality long-tail content after semantic retrieval."
        if candidate.get("is_fresh"):
            return "Fresh content matched by semantic retrieval and reranking."
        return "Matched by semantic ID transition and content reranking."

    def _rerank_reason(self, candidate: dict) -> str:
        if candidate.get("source_reason") == "long_tail_protection":
            return "long_tail_protection"
        if candidate.get("is_fresh"):
            return "freshness_boost"
        return "semantic_score_topic_author_rerank"


def create_app(data_dir: str | Path = "data/processed") -> FastAPI:
    engine = RecommendationEngine.load(data_dir)
    app = FastAPI(title="CognID-GenRec Historical Baseline")

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(request: RecommendRequest) -> RecommendResponse:
        return engine.recommend(request)

    return app
