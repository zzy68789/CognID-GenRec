from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cognid_genrec.evaluation.segmented_metrics import (
    evaluate_segmented_recommendations,
    write_segmented_metrics_report,
)
from cognid_genrec.models.baselines.itemcf import ItemCFRecommender
from cognid_genrec.models.baselines.popular import PopularRecommender
from cognid_genrec.retrieval.ann_index import NumpyANNIndex


K_VALUES = (10, 20)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KuaiRec baselines and retriever.")
    parser.add_argument("--method", choices=["popular", "itemcf", "ann_transformer"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    data_dir = Path(args.data)
    sequences = read_jsonl(data_dir / "user_sequences.jsonl")
    items = read_items(data_dir)
    recommendations = recommend(args.method, data_dir, sequences)
    report = evaluate_segmented_recommendations(
        recommendations=recommendations,
        sequences=sequences,
        item_features=items,
        k_values=K_VALUES,
    )
    output_path = Path(args.out) if args.out else Path(f"reports/metrics/kuairec_{args.method}.md")
    write_segmented_metrics_report(args.method, report, output_path)
    update_panel(args.method, report, Path("reports/metrics/kuairec_experiment_panel.md"))
    print(f"Wrote KuaiRec {args.method} metrics to {output_path}")
    print("Updated reports/metrics/kuairec_experiment_panel.md")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_items(data_dir: Path) -> pd.DataFrame:
    for filename in ["items.parquet", "item_features.parquet"]:
        path = data_dir / filename
        if path.exists():
            return pd.read_parquet(path)
    return pd.DataFrame(columns=["item_id", "topic", "hot_score"])


def recommend(method: str, data_dir: Path, sequences: list[dict]) -> dict[str, list[str]]:
    top_k = max(K_VALUES)
    histories = {
        str(sequence["user_id"]): [str(item_id) for item_id in sequence.get("train_history_item_ids", [])]
        for sequence in sequences
    }
    if method == "popular":
        model = PopularRecommender().fit(sequences)
        return {
            user_id: model.recommend(history_item_ids=history, top_k=top_k)
            for user_id, history in histories.items()
        }
    if method == "itemcf":
        model = ItemCFRecommender().fit(sequences)
        return {
            user_id: model.recommend(history_item_ids=history, top_k=top_k)
            for user_id, history in histories.items()
        }
    return ann_recommendations(data_dir, histories, top_k)


def ann_recommendations(
    data_dir: Path,
    histories: dict[str, list[str]],
    top_k: int,
) -> dict[str, list[str]]:
    embeddings = np.load(data_dir / "ann_items.npy")
    item_ids = json.loads((data_dir / "ann_item_ids.json").read_text(encoding="utf-8"))
    index = NumpyANNIndex(item_ids=item_ids, embeddings=embeddings)
    embedding_by_item = {
        str(item_id): embeddings[row_index].astype(np.float32)
        for row_index, item_id in enumerate(item_ids)
    }
    global_query = np.mean(embeddings, axis=0).astype(np.float32)
    recommendations = {}
    for user_id, history in histories.items():
        vectors = [embedding_by_item[item_id] for item_id in history if item_id in embedding_by_item]
        query = np.mean(np.stack(vectors), axis=0).astype(np.float32) if vectors else global_query
        raw = index.search(query, top_k=top_k + len(history) + 5)
        history_set = set(history)
        recommendations[user_id] = [
            item_id
            for item_id, _ in raw
            if item_id not in history_set
        ][:top_k]
    return recommendations


def update_panel(method: str, report: dict[str, dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    method_rows = read_existing_panel_methods(output_path)
    all_metrics = report.get("all", {})
    method_rows[method] = {
        "HR@10": all_metrics.get("HR@10", 0.0),
        "NDCG@10": all_metrics.get("NDCG@10", 0.0),
        "Recall@20": all_metrics.get("Recall@20", 0.0),
        "Coverage": all_metrics.get("Coverage", 0.0),
        "Diversity": all_metrics.get("Diversity", 0.0),
    }
    lines = [
        "# KuaiRec Experiment Panel",
        "",
        "Scope: offline experiments on the KuaiRec open dataset path or the built-in KuaiRec-like sample when raw files are absent.",
        "",
        "| Method | HR@10 | NDCG@10 | Recall@20 | Coverage | Diversity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method_name in sorted(method_rows):
        metrics = method_rows[method_name]
        lines.append(
            "| "
            + " | ".join(
                [
                    method_name,
                    f"{metrics.get('HR@10', 0.0):.4f}",
                    f"{metrics.get('NDCG@10', 0.0):.4f}",
                    f"{metrics.get('Recall@20', 0.0):.4f}",
                    f"{metrics.get('Coverage', 0.0):.4f}",
                    f"{metrics.get('Diversity', 0.0):.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"## Segmented Metrics ({method})",
        ]
    )
    lines.extend(
        [
        "",
        "| Segment | HR@10 | NDCG@10 | Recall@20 | Coverage | Diversity |",
        "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for segment, metrics in sorted(report.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    segment,
                    f"{metrics.get('HR@10', 0.0):.4f}",
                    f"{metrics.get('NDCG@10', 0.0):.4f}",
                    f"{metrics.get('Recall@20', 0.0):.4f}",
                    f"{metrics.get('Coverage', 0.0):.4f}",
                    f"{metrics.get('Diversity', 0.0):.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Boundary: this panel records offline results only; it is not an online CTR, watch-time, or production-system claim.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def read_existing_panel_methods(output_path: Path) -> dict[str, dict[str, float]]:
    if not output_path.exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Segmented Metrics"):
            break
        if not line.startswith("| ") or line.startswith("| Method") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 6:
            continue
        rows[cells[0]] = {
            "HR@10": float(cells[1]),
            "NDCG@10": float(cells[2]),
            "Recall@20": float(cells[3]),
            "Coverage": float(cells[4]),
            "Diversity": float(cells[5]),
        }
    return rows


if __name__ == "__main__":
    main()
