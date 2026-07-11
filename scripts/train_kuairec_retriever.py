from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from cognid_genrec.kuairec.behaviors import ACTION_TO_ID
from cognid_genrec.models.contrastive import ItemEmbeddingTable, info_nce_loss
from cognid_genrec.models.sequence_encoder import SequenceEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KuaiRec Causal Transformer retriever.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data)
    sequences = read_jsonl(data_dir / "user_sequences.jsonl")
    item_ids = collect_item_ids(data_dir, sequences)
    item_to_index = {item_id: index + 1 for index, item_id in enumerate(item_ids)}
    dataset = build_dataset(sequences, item_to_index, args.max_len)
    if len(dataset.tensors[0]) == 0:
        raise ValueError("no training examples; run prepare_kuairec_data with enough history")

    encoder = SequenceEncoder(
        num_items=len(item_to_index),
        num_actions=max(ACTION_TO_ID.values()),
        hidden_dim=args.hidden_dim,
        max_len=args.max_len,
    )
    item_embeddings = ItemEmbeddingTable(num_items=len(item_to_index), hidden_dim=args.hidden_dim)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *item_embeddings.parameters()],
        lr=1e-3,
        weight_decay=0.01,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    losses: list[float] = []
    batch_count = 0
    encoder.train()
    item_embeddings.train()
    for _ in range(args.epochs):
        for item_batch, action_batch, delta_batch, target_batch in loader:
            user_vectors = encoder(item_batch, action_batch, delta_batch)
            positive_vectors = item_embeddings(target_batch)
            loss = info_nce_loss(user_vectors, positive_vectors)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            batch_count += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "item_embedding_state_dict": item_embeddings.state_dict(),
            "item_to_index": item_to_index,
            "action_to_id": ACTION_TO_ID,
            "config": {
                "hidden_dim": args.hidden_dim,
                "max_len": args.max_len,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
            },
        },
        out_path,
    )
    ann_vectors = item_embeddings.embedding.weight.detach().cpu().numpy()[1:].astype(np.float32)
    np.save(data_dir / "ann_items.npy", ann_vectors)
    (data_dir / "ann_item_ids.json").write_text(
        json.dumps(item_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = Path("reports/metrics/kuairec_retriever_train.md")
    write_training_report(
        report_path,
        {
            "loss": losses[-1] if losses else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "batch_count": float(batch_count),
            "item_count": float(len(item_ids)),
            "user_sequence_count": float(len(sequences)),
            "hidden_dim": float(args.hidden_dim),
            "max_len": float(args.max_len),
        },
    )
    print(f"Wrote KuaiRec retriever checkpoint to {out_path}")
    print(f"Wrote ANN artifacts to {data_dir / 'ann_items.npy'} and {data_dir / 'ann_item_ids.json'}")
    print(f"Wrote training report to {report_path}")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collect_item_ids(data_dir: Path, sequences: list[dict]) -> list[str]:
    item_ids: set[str] = set()
    items_path = data_dir / "items.parquet"
    if items_path.exists():
        items = pd.read_parquet(items_path)
        item_ids |= set(items["item_id"].astype(str))
    for sequence in sequences:
        item_ids |= {str(item_id) for item_id in sequence.get("item_ids", [])}
    return sorted(item_ids, key=lambda value: (0, int(value)) if value.isdigit() else (1, value))


def build_dataset(
    sequences: list[dict],
    item_to_index: dict[str, int],
    max_len: int,
) -> TensorDataset:
    item_rows = []
    action_rows = []
    delta_rows = []
    targets = []
    for sequence in sequences:
        history = [str(item_id) for item_id in sequence.get("train_history_item_ids", [])][-max_len:]
        target = str(sequence.get("validation_item_id") or sequence.get("test_item_id"))
        if not history or target not in item_to_index:
            continue
        history_length = len(history)
        actions = sequence.get("actions", [])[:history_length]
        deltas = sequence.get("time_deltas", [])[:history_length]
        item_row = [item_to_index.get(item_id, 0) for item_id in history]
        action_row = [ACTION_TO_ID.get(str(action), 0) for action in actions]
        delta_row = [float(delta) for delta in deltas]
        item_row, action_row, delta_row = pad_rows(item_row, action_row, delta_row, max_len)
        item_rows.append(item_row)
        action_rows.append(action_row)
        delta_rows.append(delta_row)
        targets.append(item_to_index[target])
    return TensorDataset(
        torch.tensor(item_rows, dtype=torch.long),
        torch.tensor(action_rows, dtype=torch.long),
        torch.tensor(delta_rows, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
    )


def pad_rows(
    item_row: list[int],
    action_row: list[int],
    delta_row: list[float],
    max_len: int,
) -> tuple[list[int], list[int], list[float]]:
    item_row = item_row[-max_len:]
    action_row = action_row[-max_len:]
    delta_row = delta_row[-max_len:]
    pad_count = max_len - len(item_row)
    return (
        item_row + [0] * pad_count,
        action_row + [0] * pad_count,
        delta_row + [0.0] * pad_count,
    )


def write_training_report(output_path: Path, metrics: dict[str, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KuaiRec Retriever Training",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for metric_name, value in metrics.items():
        lines.append(f"| {metric_name} | {value:.4f} |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
