from __future__ import annotations

import json
from pathlib import Path


def write_metrics_report(method: str, metrics: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Retrieval Evaluation: {method}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for metric_name, value in metrics.items():
        lines.append(f"| {metric_name} | {value:.4f} |")
    lines.extend(
        [
            "",
            "```json",
            json.dumps(metrics, sort_keys=True),
            "```",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def update_baseline_summary(method: str, metrics: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_sections = _read_baseline_sections(output_path)
    existing_sections[method] = metrics

    lines = [
        "# Baseline Retrieval Metrics",
        "",
        "| Method | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | Coverage | Diversity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method_name in sorted(existing_sections):
        method_metrics = existing_sections[method_name]
        lines.append(
            "| "
            + " | ".join(
                [
                    method_name,
                    f"{method_metrics.get('Recall@5', 0.0):.4f}",
                    f"{method_metrics.get('Recall@10', 0.0):.4f}",
                    f"{method_metrics.get('NDCG@5', 0.0):.4f}",
                    f"{method_metrics.get('NDCG@10', 0.0):.4f}",
                    f"{method_metrics.get('Coverage', 0.0):.4f}",
                    f"{method_metrics.get('Diversity', 0.0):.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _read_baseline_sections(output_path: Path) -> dict[str, dict[str, float]]:
    if not output_path.exists():
        return {}
    sections: dict[str, dict[str, float]] = {}
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| Method") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 7:
            continue
        method, recall5, recall10, ndcg5, ndcg10, coverage, diversity = cells
        sections[method] = {
            "Recall@5": float(recall5),
            "Recall@10": float(recall10),
            "NDCG@5": float(ndcg5),
            "NDCG@10": float(ndcg10),
            "Coverage": float(coverage),
            "Diversity": float(diversity),
        }
    return sections
