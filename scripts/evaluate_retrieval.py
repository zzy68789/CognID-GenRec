from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cognid_genrec.evaluation.reports import (
    update_baseline_summary,
    write_metrics_report,
)
from cognid_genrec.evaluation.ranking_metrics import evaluate_recommendations
from cognid_genrec.models.generative_retriever import GenerativeRetriever
from cognid_genrec.models.baselines.itemcf import ItemCFRecommender
from cognid_genrec.models.baselines.popular import PopularRecommender
from cognid_genrec.retrieval.diversity_reranker import DiversityReranker


K_VALUES = (5, 10)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval baselines.")
    parser.add_argument(
        "--method",
        choices=["popular", "itemcf", "generative", "generative_rerank"],
        required=True,
    )
    parser.add_argument("--data", required=True, help="Processed data directory.")
    parser.add_argument("--out", required=True, help="Markdown report output path.")
    args = parser.parse_args()

    data_dir = Path(args.data)
    metrics = evaluate_method(args.method, data_dir)
    output_path = Path(args.out)
    write_metrics_report(args.method, metrics, output_path)
    update_baseline_summary(args.method, metrics, output_path.parent / "baseline_metrics.md")
    print(f"Wrote {args.method} metrics to {output_path}")


def evaluate_method(method: str, data_dir: Path) -> dict[str, float]:
    sequences = read_jsonl(data_dir / "user_sequences.jsonl")
    items = load_items(data_dir / "items.csv")
    item_topics = dict(zip(items["item_id"].astype(str), items["topic"].astype(str), strict=False))
    all_item_ids = set(item_topics) | {
        item_id
        for sequence in sequences
        for item_id in sequence.get("item_ids", [])
    }
    targets = {
        sequence["user_id"]: sequence["test_item_id"]
        for sequence in sequences
        if sequence.get("test_item_id")
    }
    histories = {
        sequence["user_id"]: sequence.get("train_history_item_ids", [])
        for sequence in sequences
    }

    recommender = build_recommender(method, sequences, data_dir)
    top_k = max(K_VALUES)
    if method == "generative_rerank":
        reranker = DiversityReranker()
        recommendations = {}
        for user_id, history in histories.items():
            raw_items = recommender.recommend(history_item_ids=history, top_k=top_k * 3)
            candidates = [
                {"item_id": item_id, "score": 1.0 / (rank + 1)}
                for rank, item_id in enumerate(raw_items)
            ]
            recommendations[user_id] = [
                candidate["item_id"]
                for candidate in reranker.rerank(candidates, items, top_k=top_k)
            ]
    else:
        recommendations = {
            user_id: recommender.recommend(history_item_ids=history, top_k=top_k)
            for user_id, history in histories.items()
        }
    return evaluate_recommendations(
        recommendations=recommendations,
        targets=targets,
        all_item_ids=all_item_ids,
        item_topics=item_topics,
        k_values=K_VALUES,
    )


def build_recommender(method: str, sequences: list[dict], data_dir: Path):
    if method == "popular":
        return PopularRecommender().fit(sequences)
    if method == "itemcf":
        return ItemCFRecommender().fit(sequences)
    if method in {"generative", "generative_rerank"}:
        artifact_path = data_dir / "generative_retriever.json"
        if artifact_path.exists():
            return GenerativeRetriever.load(artifact_path)
        semantic_mapping = pd.read_parquet(data_dir / "semantic_ids.parquet")
        return GenerativeRetriever().fit(sequences, semantic_mapping)
    raise ValueError(f"unsupported method: {method}")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_items(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["item_id", "topic", "author_id", "publish_time", "quality_score"])
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
