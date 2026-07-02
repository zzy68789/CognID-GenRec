from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd


def hit_rate_at_k(recommended: list[str], target: str, k: int) -> float:
    return 1.0 if str(target) in [str(item) for item in recommended[:k]] else 0.0


def weighted_hit_rate_at_k(
    recommended: list[str],
    target: str,
    target_weight: float,
    k: int,
) -> float:
    return float(target_weight) * hit_rate_at_k(recommended, target, k)


def evaluate_segmented_recommendations(
    recommendations: Mapping[str, Sequence[str]],
    sequences: Sequence[dict],
    item_features: pd.DataFrame,
    k_values: Sequence[int] = (10, 20),
) -> dict[str, dict[str, float]]:
    item_topics = _item_topics(item_features)
    item_popularity = _item_popularity(item_features)
    catalog = set(item_features.get("item_id", pd.Series(dtype=str)).astype(str))
    if not catalog:
        catalog = {
            str(item_id)
            for ranked_items in recommendations.values()
            for item_id in ranked_items
        } | {
            str(sequence.get("test_item_id"))
            for sequence in sequences
            if sequence.get("test_item_id")
        }

    groups: dict[str, list[dict]] = {"all": list(sequences)}
    for sequence in sequences:
        action = str(sequence.get("test_action", "unknown"))
        groups.setdefault(f"action={action}", []).append(sequence)
        activity = _activity_bucket(len(sequence.get("train_history_item_ids", [])))
        groups.setdefault(f"activity={activity}", []).append(sequence)
        target = str(sequence.get("test_item_id", ""))
        groups.setdefault(f"popularity={item_popularity.get(target, 'tail')}", []).append(sequence)

    return {
        segment_name: _metrics_for_group(
            recommendations=recommendations,
            sequences=segment_sequences,
            item_topics=item_topics,
            catalog=catalog,
            k_values=k_values,
        )
        for segment_name, segment_sequences in groups.items()
    }


def write_segmented_metrics_report(
    method: str,
    report: dict[str, dict[str, float]],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = sorted({metric for metrics in report.values() for metric in metrics})
    lines = [
        f"# KuaiRec Retrieval Evaluation: {method}",
        "",
        "| Segment | " + " | ".join(metric_names) + " |",
        "|---|" + "|".join(["---:"] * len(metric_names)) + "|",
    ]
    for segment_name in sorted(report):
        values = [f"{report[segment_name].get(metric, 0.0):.4f}" for metric in metric_names]
        lines.append("| " + " | ".join([segment_name, *values]) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _metrics_for_group(
    recommendations: Mapping[str, Sequence[str]],
    sequences: Sequence[dict],
    item_topics: Mapping[str, str],
    catalog: set[str],
    k_values: Sequence[int],
) -> dict[str, float]:
    if not sequences:
        return {}
    metrics: dict[str, float] = {}
    for k in k_values:
        hits = 0.0
        ndcg = 0.0
        recall = 0.0
        for sequence in sequences:
            user_id = str(sequence["user_id"])
            target = str(sequence["test_item_id"])
            ranked = [str(item_id) for item_id in recommendations.get(user_id, [])]
            hit = hit_rate_at_k(ranked, target, k)
            hits += hit
            recall += hit
            if hit:
                rank_index = ranked[:k].index(target)
                ndcg += 1.0 / math.log2(rank_index + 2)
        denominator = len(sequences)
        metrics[f"HR@{k}"] = hits / denominator
        metrics[f"NDCG@{k}"] = ndcg / denominator
        metrics[f"Recall@{k}"] = recall / denominator

    max_k = max(k_values) if k_values else 10
    recommended_items = {
        str(item_id)
        for sequence in sequences
        for item_id in recommendations.get(str(sequence["user_id"]), [])[:max_k]
    }
    metrics["Coverage"] = len(recommended_items & catalog) / len(catalog) if catalog else 0.0
    metrics["Diversity"] = _diversity_for_group(recommendations, sequences, item_topics, max_k)
    return metrics


def _diversity_for_group(
    recommendations: Mapping[str, Sequence[str]],
    sequences: Sequence[dict],
    item_topics: Mapping[str, str],
    k: int,
) -> float:
    scores = []
    for sequence in sequences:
        ranked = [str(item_id) for item_id in recommendations.get(str(sequence["user_id"]), [])[:k]]
        if not ranked:
            continue
        topics = {item_topics.get(item_id, "unknown") for item_id in ranked}
        scores.append(len(topics) / len(ranked))
    return sum(scores) / len(scores) if scores else 0.0


def _item_topics(item_features: pd.DataFrame) -> dict[str, str]:
    if item_features.empty or "item_id" not in item_features.columns:
        return {}
    topic = item_features["topic"] if "topic" in item_features.columns else "unknown"
    return dict(zip(item_features["item_id"].astype(str), pd.Series(topic).astype(str), strict=False))


def _item_popularity(item_features: pd.DataFrame) -> dict[str, str]:
    if item_features.empty or "item_id" not in item_features.columns or "hot_score" not in item_features.columns:
        return {}
    scores = pd.to_numeric(item_features["hot_score"], errors="coerce").fillna(0.0)
    high = scores.quantile(0.66)
    low = scores.quantile(0.33)
    buckets = []
    for score in scores:
        if score >= high:
            buckets.append("head")
        elif score <= low:
            buckets.append("tail")
        else:
            buckets.append("middle")
    return dict(zip(item_features["item_id"].astype(str), buckets, strict=False))


def _activity_bucket(length: int) -> str:
    if length < 2:
        return "short"
    if length < 10:
        return "medium"
    return "long"
