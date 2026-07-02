from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cognid_genrec.models.generative_retriever import GenerativeRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight generative retriever.")
    parser.add_argument("--data", required=True, help="Processed data directory.")
    parser.add_argument("--semantic-ids", required=True, help="Semantic ID parquet path.")
    parser.add_argument("--out", required=True, help="Training report markdown path.")
    args = parser.parse_args()

    data_dir = Path(args.data)
    sequences = read_jsonl(data_dir / "user_sequences.jsonl")
    semantic_mapping = pd.read_parquet(args.semantic_ids)
    retriever = GenerativeRetriever().fit(sequences, semantic_mapping)

    artifact_path = data_dir / "generative_retriever.json"
    retriever.save(artifact_path)
    metrics = {
        "sequence_count": float(len(sequences)),
        "semantic_id_count": float(len(semantic_mapping)),
        "transition_count": float(
            sum(len(targets) for targets in retriever.transition_counts.values())
        ),
        "artifact_path": str(artifact_path),
    }
    write_training_report(metrics, Path(args.out))
    print(f"Wrote generative retriever artifact to {artifact_path}")
    print(f"Wrote training report to {args.out}")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_training_report(metrics: dict[str, float | str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generative Retriever Training",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            value_text = f"{value:.4f}"
        else:
            value_text = str(value)
        lines.append(f"| {metric_name} | {value_text} |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
