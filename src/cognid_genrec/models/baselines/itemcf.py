from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import combinations


class ItemCFRecommender:
    def __init__(self, max_items_per_user: int = 100, max_neighbors_per_item: int = 200) -> None:
        self.max_items_per_user = max_items_per_user
        self.max_neighbors_per_item = max_neighbors_per_item
        self.item_similarities: dict[str, dict[str, float]] = {}
        self.item_popularity: Counter[str] = Counter()

    def fit(self, user_sequences: Iterable[dict]) -> "ItemCFRecommender":
        similarities: defaultdict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        popularity: Counter[str] = Counter()

        for sequence in user_sequences:
            item_ids = _deduplicate_preserving_order(
                str(item_id) for item_id in sequence.get("item_ids", [])
            )
            unique_items = item_ids[-self.max_items_per_user :]
            popularity.update(unique_items)
            for left, right in combinations(unique_items, 2):
                similarities[left][right] += 1.0
                similarities[right][left] += 1.0

        self.item_similarities = {
            item_id: dict(_top_neighbors(neighbors, self.max_neighbors_per_item))
            for item_id, neighbors in similarities.items()
        }
        self.item_popularity = popularity
        return self

    def recommend(self, history_item_ids: Iterable[str] | None = None, top_k: int = 10) -> list[str]:
        history = {str(item_id) for item_id in history_item_ids or []}
        scores: defaultdict[str, float] = defaultdict(float)
        for item_id in history:
            for candidate, similarity in self.item_similarities.get(item_id, {}).items():
                if candidate in history:
                    continue
                scores[candidate] += similarity

        if not scores:
            for item_id, count in self.item_popularity.items():
                if item_id not in history:
                    scores[item_id] = float(count)

        ranked = sorted(scores.items(), key=lambda item_score: (-item_score[1], item_score[0]))
        return [item_id for item_id, _ in ranked[:top_k]]


def _deduplicate_preserving_order(item_ids: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for item_id in item_ids:
        if item_id in seen:
            continue
        seen.add(item_id)
        result.append(item_id)
    return result


def _top_neighbors(neighbors: dict[str, float], limit: int) -> list[tuple[str, float]]:
    if limit <= 0:
        return []
    return sorted(neighbors.items(), key=lambda item_score: (-item_score[1], item_score[0]))[:limit]
