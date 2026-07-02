from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def build_kuairec_sequences(
    frame: pd.DataFrame,
    min_history: int = 3,
    min_action_weight: float = 0.0,
) -> list[dict]:
    required = {"user_id", "video_id", "timestamp", "action", "action_weight"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"interactions missing columns: {sorted(missing)}")

    filtered = frame[pd.to_numeric(frame["action_weight"], errors="coerce") >= min_action_weight]
    sorted_frame = filtered.sort_values(["user_id", "timestamp", "video_id"])
    sequences = []
    for user_id, user_events in sorted_frame.groupby("user_id", sort=True):
        if len(user_events) < min_history + 2:
            continue
        timestamps = [int(value) for value in user_events["timestamp"].tolist()]
        deltas = [0, *[max(0, right - left) for left, right in zip(timestamps, timestamps[1:])]]
        item_ids = [str(value) for value in user_events["video_id"].tolist()]
        actions = [str(value) for value in user_events["action"].tolist()]
        weights = [float(value) for value in user_events["action_weight"].tolist()]
        sequences.append(
            {
                "user_id": str(user_id),
                "item_ids": item_ids,
                "actions": actions,
                "action_weights": weights,
                "event_weights": weights,
                "timestamps": timestamps,
                "time_deltas": deltas,
                "train_history_item_ids": item_ids[:-2],
                "validation_item_id": item_ids[-2],
                "test_item_id": item_ids[-1],
                "validation_action": actions[-2],
                "test_action": actions[-1],
                "validation_action_weight": weights[-2],
                "test_action_weight": weights[-1],
                "sequence_length": len(item_ids),
            }
        )
    return sequences


def write_kuairec_sequences(sequences: Iterable[dict], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sequence in sequences:
            handle.write(json.dumps(sequence, ensure_ascii=False) + "\n")


def read_kuairec_sequences(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
