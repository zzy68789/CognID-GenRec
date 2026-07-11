from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cognid_genrec.evaluation.protocol import (
    CandidateManifest,
    build_train_only_sequences,
    evaluation_histories,
    read_candidate_manifest,
    validate_recommendations,
    validate_sequential_protocol,
)
from cognid_genrec.evaluation.segmented_metrics import (
    evaluate_segmented_recommendations,
    write_segmented_metrics_report,
)
from cognid_genrec.models.baselines.itemcf import ItemCFRecommender
from cognid_genrec.models.baselines.popular import PopularRecommender
from cognid_genrec.retrieval.ann_index import NumpyANNIndex


K_VALUES = (10, 20)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate KuaiRec baselines and retriever."
    )
    parser.add_argument(
        "--method", choices=["popular", "itemcf", "ann_transformer"], required=True
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--candidate-manifest", default="")
    args = parser.parse_args()

    data_dir = Path(args.data)
    sequences = read_jsonl(data_dir / "user_sequences.jsonl")
    items = read_items(data_dir)
    manifest_path = (
        Path(args.candidate_manifest)
        if args.candidate_manifest
        else data_dir / "candidate_manifest.json"
    )
    manifest = read_candidate_manifest(manifest_path)
    validate_sequential_protocol(sequences, manifest)
    recommendations = recommend(args.method, data_dir, sequences, manifest)
    report = evaluate_segmented_recommendations(
        recommendations=recommendations,
        sequences=sequences,
        item_features=items,
        k_values=K_VALUES,
        candidate_item_ids=manifest.candidate_item_ids,
    )
    output_path = (
        Path(args.out)
        if args.out
        else Path(f"reports/metrics/kuairec_{manifest.protocol_name}_{args.method}.md")
    )
    metadata = {
        "protocol": manifest.protocol_name,
        "dataset_matrix": manifest.dataset_matrix,
        "ranking_mode": manifest.ranking_mode,
        "candidate_source": manifest.candidate_source,
        "candidate_count": manifest.candidate_count,
        "fit_boundary": manifest.fit_history_field,
        "history_field": manifest.history_field,
        "target_field": manifest.target_field,
        "random_seed": manifest.random_seed,
    }
    write_segmented_metrics_report(args.method, report, output_path, metadata=metadata)
    method_label = f"{manifest.protocol_name}/{args.method}"
    panel_path = Path(
        f"reports/metrics/kuairec_{manifest.protocol_name}_experiment_panel.md"
    )
    update_panel(
        method_label,
        report,
        panel_path,
        protocol_name=manifest.protocol_name,
        candidate_count=manifest.candidate_count,
    )
    print(f"Wrote KuaiRec {args.method} metrics to {output_path}")
    print(f"Updated {panel_path}")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_items(data_dir: Path) -> pd.DataFrame:
    for filename in ["items.parquet", "item_features.parquet"]:
        path = data_dir / filename
        if path.exists():
            return pd.read_parquet(path)
    return pd.DataFrame(columns=["item_id", "topic", "hot_score"])


def recommend(
    method: str,
    data_dir: Path,
    sequences: list[dict],
    manifest: CandidateManifest,
    top_k: int | None = None,
) -> dict[str, list[str]]:
    top_k = top_k or max(K_VALUES)
    histories = evaluation_histories(sequences, manifest)
    train_sequences = build_train_only_sequences(sequences)
    candidates = manifest.candidate_item_ids
    if method == "popular":
        model = PopularRecommender().fit(train_sequences)
        recommendations = {
            user_id: model.recommend(
                history_item_ids=history,
                top_k=top_k,
                candidate_item_ids=candidates,
            )
            for user_id, history in histories.items()
        }
    elif method == "itemcf":
        model = ItemCFRecommender().fit(train_sequences)
        recommendations = {
            user_id: model.recommend(
                history_item_ids=history,
                top_k=top_k,
                candidate_item_ids=candidates,
            )
            for user_id, history in histories.items()
        }
    else:
        recommendations = ann_recommendations(data_dir, histories, candidates, top_k)
    validate_recommendations(recommendations, sequences, manifest, top_k)
    return recommendations


def ann_recommendations(
    data_dir: Path,
    histories: dict[str, list[str]],
    candidate_item_ids: tuple[str, ...],
    top_k: int,
) -> dict[str, list[str]]:
    embeddings = np.load(data_dir / "ann_items.npy")
    item_ids = json.loads((data_dir / "ann_item_ids.json").read_text(encoding="utf-8"))
    embedding_by_item = {
        str(item_id): embeddings[row_index].astype(np.float32)
        for row_index, item_id in enumerate(item_ids)
    }
    missing_candidates = [
        item_id for item_id in candidate_item_ids if item_id not in embedding_by_item
    ]
    if missing_candidates:
        raise ValueError(
            f"ANN artifact is missing {len(missing_candidates)} protocol candidates; "
            f"examples={missing_candidates[:5]}"
        )
    candidate_embeddings = np.stack(
        [embedding_by_item[item_id] for item_id in candidate_item_ids]
    ).astype(np.float32)
    index = NumpyANNIndex(
        item_ids=list(candidate_item_ids),
        embeddings=candidate_embeddings,
    )
    global_query = np.mean(candidate_embeddings, axis=0).astype(np.float32)
    recommendations = {}
    candidate_set = set(candidate_item_ids)
    for user_id, history in histories.items():
        vectors = [
            embedding_by_item[item_id]
            for item_id in history
            if item_id in embedding_by_item
        ]
        query = (
            np.mean(np.stack(vectors), axis=0).astype(np.float32)
            if vectors
            else global_query
        )
        history_in_candidates = len(set(history) & candidate_set)
        raw = index.search(
            query, top_k=min(len(candidate_item_ids), top_k + history_in_candidates)
        )
        history_set = set(history)
        recommendations[user_id] = [
            item_id for item_id, _ in raw if item_id not in history_set
        ][:top_k]
    return recommendations


def update_panel(
    method: str,
    report: dict[str, dict[str, float]],
    output_path: Path,
    protocol_name: str = "legacy_diagnostic",
    candidate_count: int = 0,
) -> None:
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
        f"Protocol: {protocol_name}; ranking_mode=full_sort; candidate_count={candidate_count}.",
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
        if (
            not line.startswith("| ")
            or line.startswith("| Method")
            or line.startswith("|---")
        ):
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
