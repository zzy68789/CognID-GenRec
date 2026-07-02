from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


POSITIVE_EVENT_WEIGHTS = {
    "click": 1.0,
    "like": 2.0,
    "collect": 3.0,
}


def build_user_sequences(interactions: pd.DataFrame) -> list[dict]:
    positive = interactions[interactions["event_type"].isin(POSITIVE_EVENT_WEIGHTS)].copy()
    if positive.empty:
        return []

    positive["event_time"] = pd.to_datetime(positive["event_time"], errors="raise")
    positive = positive.sort_values(["user_id", "event_time", "item_id"])

    sequences: list[dict] = []
    for user_id, user_events in positive.groupby("user_id", sort=True):
        if len(user_events) < 3:
            continue

        item_ids = user_events["item_id"].astype(str).tolist()
        event_weights = [
            POSITIVE_EVENT_WEIGHTS[event_type]
            for event_type in user_events["event_type"].astype(str).tolist()
        ]
        sequences.append(
            {
                "user_id": str(user_id),
                "item_ids": item_ids,
                "event_weights": event_weights,
                "train_history_item_ids": item_ids[:-2],
                "validation_item_id": item_ids[-2],
                "test_item_id": item_ids[-1],
            }
        )

    return sequences


def write_user_sequences(sequences: Iterable[dict], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sequence in sequences:
            handle.write(json.dumps(sequence, ensure_ascii=False) + "\n")
