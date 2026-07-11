from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import pandas as pd


EVENT_WEIGHTS = {
    "click": 1.0,
    "like": 2.0,
    "collect": 3.0,
}


class PopularRecommender:
    def __init__(self) -> None:
        self.item_scores: dict[str, float] = {}

    def fit(
        self, interactions_or_sequences: pd.DataFrame | Iterable[dict]
    ) -> "PopularRecommender":
        scores: defaultdict[str, float] = defaultdict(float)
        if isinstance(interactions_or_sequences, pd.DataFrame):
            for row in interactions_or_sequences.to_dict("records"):
                item_id = row.get("item_id", row.get("video_id"))
                if item_id is None:
                    continue
                if "action_weight" in row:
                    scores[str(item_id)] += float(row["action_weight"])
                    continue
                event_type = str(row.get("event_type", ""))
                if event_type in EVENT_WEIGHTS:
                    scores[str(item_id)] += EVENT_WEIGHTS[event_type]
        else:
            for sequence in interactions_or_sequences:
                item_ids = [str(item_id) for item_id in sequence.get("item_ids", [])]
                event_weights = (
                    sequence.get("event_weights")
                    or sequence.get("action_weights")
                    or [1.0] * len(item_ids)
                )
                for item_id, weight in zip(item_ids, event_weights, strict=False):
                    scores[item_id] += float(weight)

        self.item_scores = dict(scores)
        return self

    def recommend(
        self,
        history_item_ids: Iterable[str] | None = None,
        top_k: int = 10,
        candidate_item_ids: Iterable[str] | None = None,
    ) -> list[str]:
        history = set(history_item_ids or [])
        candidates = (
            {str(item_id) for item_id in candidate_item_ids}
            if candidate_item_ids is not None
            else set(self.item_scores)
        )
        ranked = sorted(
            (
                (item_id, self.item_scores.get(item_id, 0.0))
                for item_id in candidates
                if item_id not in history
            ),
            key=lambda item_score: (-item_score[1], item_score[0]),
        )
        return [item_id for item_id, _ in ranked[:top_k]]
