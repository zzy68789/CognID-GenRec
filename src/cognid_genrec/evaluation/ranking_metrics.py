from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def recall_at_k(
    recommendations: Mapping[str, Sequence[str]],
    targets: Mapping[str, str],
    k: int,
) -> float:
    if not targets:
        return 0.0
    hits = sum(
        1
        for user_id, target_item in targets.items()
        if target_item in recommendations.get(user_id, [])[:k]
    )
    return hits / len(targets)


def ndcg_at_k(
    recommendations: Mapping[str, Sequence[str]],
    targets: Mapping[str, str],
    k: int,
) -> float:
    if not targets:
        return 0.0

    total = 0.0
    for user_id, target_item in targets.items():
        ranked_items = recommendations.get(user_id, [])[:k]
        if target_item not in ranked_items:
            continue
        rank_index = ranked_items.index(target_item)
        total += 1.0 / math.log2(rank_index + 2)
    return total / len(targets)


def coverage_at_k(
    recommendations: Mapping[str, Sequence[str]],
    all_item_ids: Iterable[str],
    k: int,
) -> float:
    catalog = set(all_item_ids)
    if not catalog:
        return 0.0
    recommended = {
        item_id
        for ranked_items in recommendations.values()
        for item_id in ranked_items[:k]
    }
    return len(recommended & catalog) / len(catalog)


def topic_diversity_at_k(
    recommendations: Mapping[str, Sequence[str]],
    item_topics: Mapping[str, str],
    k: int,
) -> float:
    per_user_scores: list[float] = []
    for ranked_items in recommendations.values():
        top_items = ranked_items[:k]
        if not top_items:
            continue
        topics = {item_topics[item_id] for item_id in top_items if item_id in item_topics}
        if not topics:
            per_user_scores.append(0.0)
            continue
        per_user_scores.append(len(topics) / len(top_items))
    if not per_user_scores:
        return 0.0
    return sum(per_user_scores) / len(per_user_scores)


def evaluate_recommendations(
    recommendations: Mapping[str, Sequence[str]],
    targets: Mapping[str, str],
    all_item_ids: Iterable[str],
    item_topics: Mapping[str, str],
    k_values: Sequence[int] = (5, 10),
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"Recall@{k}"] = recall_at_k(recommendations, targets, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(recommendations, targets, k)
    max_k = max(k_values) if k_values else 10
    metrics["Coverage"] = coverage_at_k(recommendations, all_item_ids, max_k)
    metrics["Diversity"] = topic_diversity_at_k(recommendations, item_topics, max_k)
    return metrics
