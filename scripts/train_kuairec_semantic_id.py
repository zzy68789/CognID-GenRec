from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cognid_genrec.tokenizer.semantic_id_tokenizer import (
    RQVAESemanticIDTokenizer,
    SemanticIDTokenizer,
    collision_rate,
    semantic_id_quality_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KuaiRec short-video semantic IDs.")
    parser.add_argument("--items", required=True)
    parser.add_argument("--method", choices=["kmeans", "rqvae"], default="kmeans")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-iter", type=int, default=20)
    args = parser.parse_args()

    items = read_items(Path(args.items))
    tokenizer = build_tokenizer(args.method, args.max_iter)
    mapping = tokenizer.fit_transform(items)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(output_path, index=False)

    metrics = semantic_id_quality_metrics(mapping)
    metrics.update(bucket_collision_metrics(mapping, items))
    report_path = Path(f"reports/metrics/kuairec_semantic_id_{args.method}.md")
    write_report(args.method, metrics, report_path)
    print(f"Wrote KuaiRec semantic IDs for {len(mapping)} items to {output_path}")
    print(f"Wrote KuaiRec semantic ID report to {report_path}")


def read_items(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported item file type: {path.suffix}")


def build_tokenizer(method: str, max_iter: int):
    if method == "kmeans":
        return SemanticIDTokenizer()
    return RQVAESemanticIDTokenizer(max_iter=max_iter)


def bucket_collision_metrics(mapping: pd.DataFrame, items: pd.DataFrame) -> dict[str, float]:
    if "hot_score" not in items.columns:
        return {
            "collision_rate_head": 0.0,
            "collision_rate_middle": 0.0,
            "collision_rate_tail": 0.0,
        }
    merged = mapping.merge(items[["item_id", "hot_score"]], on="item_id", how="left")
    scores = pd.to_numeric(merged["hot_score"], errors="coerce").fillna(0.0)
    high = scores.quantile(0.66)
    low = scores.quantile(0.33)
    buckets = {
        "head": merged[scores >= high],
        "middle": merged[(scores < high) & (scores > low)],
        "tail": merged[scores <= low],
    }
    return {
        f"collision_rate_{bucket}": collision_rate(frame)
        for bucket, frame in buckets.items()
    }


def write_report(method: str, metrics: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# KuaiRec Semantic ID Quality: {method}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for metric_name, value in metrics.items():
        lines.append(f"| {metric_name} | {float(value):.4f} |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
