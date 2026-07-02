from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cognid_genrec.tokenizer.semantic_id_tokenizer import (
    RQVAESemanticIDTokenizer,
    SemanticIDTokenizer,
    semantic_id_quality_metrics,
    write_semantic_id_quality_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train semantic ID tokenizer.")
    parser.add_argument("--items", required=True, help="Path to processed items parquet or CSV.")
    parser.add_argument("--method", choices=["kmeans", "rqvae"], default="kmeans")
    parser.add_argument("--out", required=True, help="Output semantic ID parquet path.")
    args = parser.parse_args()

    items = read_items(Path(args.items))
    tokenizer = build_tokenizer(args.method)
    mapping = tokenizer.fit_transform(items)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(output_path, index=False)

    metrics = semantic_id_quality_metrics(mapping)
    report_path = default_report_path(args.method)
    write_semantic_id_quality_report(metrics, report_path)
    print(f"Wrote semantic IDs for {len(mapping)} items to {output_path}")
    print(f"Wrote semantic ID quality report to {report_path}")


def read_items(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported item file type: {path.suffix}")


def build_tokenizer(method: str):
    if method == "kmeans":
        return SemanticIDTokenizer()
    if method == "rqvae":
        return RQVAESemanticIDTokenizer()
    raise ValueError(f"unsupported semantic ID method: {method}")


def default_report_path(method: str) -> Path:
    if method == "kmeans":
        return Path("reports/metrics/semantic_id_quality.md")
    return Path(f"reports/metrics/semantic_id_quality_{method}.md")


if __name__ == "__main__":
    main()
