import json

import pandas as pd


def _sequence_builder():
    from cognid_genrec.data.sequence_builder import (
        build_user_sequences,
        write_user_sequences,
    )

    return build_user_sequences, write_user_sequences


def test_build_user_sequences_sorts_filters_skip_and_splits():
    build_user_sequences, _ = _sequence_builder()
    interactions = pd.DataFrame(
        [
            {
                "user_id": "u_001",
                "item_id": "i_003",
                "event_type": "collect",
                "event_time": "2026-06-01T10:03:00",
                "dwell_time": 60.0,
            },
            {
                "user_id": "u_001",
                "item_id": "i_001",
                "event_type": "click",
                "event_time": "2026-06-01T10:01:00",
                "dwell_time": 12.0,
            },
            {
                "user_id": "u_001",
                "item_id": "i_999",
                "event_type": "skip",
                "event_time": "2026-06-01T10:02:00",
                "dwell_time": 1.0,
            },
            {
                "user_id": "u_001",
                "item_id": "i_002",
                "event_type": "like",
                "event_time": "2026-06-01T10:02:30",
                "dwell_time": 35.0,
            },
            {
                "user_id": "u_001",
                "item_id": "i_004",
                "event_type": "click",
                "event_time": "2026-06-01T10:04:00",
                "dwell_time": 15.0,
            },
        ]
    )

    sequences = build_user_sequences(interactions)

    assert sequences == [
        {
            "user_id": "u_001",
            "item_ids": ["i_001", "i_002", "i_003", "i_004"],
            "event_weights": [1.0, 2.0, 3.0, 1.0],
            "train_history_item_ids": ["i_001", "i_002"],
            "validation_item_id": "i_003",
            "test_item_id": "i_004",
        }
    ]


def test_build_user_sequences_skips_users_without_enough_positive_events():
    build_user_sequences, _ = _sequence_builder()
    interactions = pd.DataFrame(
        [
            {
                "user_id": "u_001",
                "item_id": "i_001",
                "event_type": "click",
                "event_time": "2026-06-01T10:01:00",
                "dwell_time": 12.0,
            },
            {
                "user_id": "u_001",
                "item_id": "i_002",
                "event_type": "skip",
                "event_time": "2026-06-01T10:02:00",
                "dwell_time": 1.0,
            },
        ]
    )

    assert build_user_sequences(interactions) == []


def test_write_user_sequences_outputs_jsonl(tmp_path):
    _, write_user_sequences = _sequence_builder()
    output_path = tmp_path / "user_sequences.jsonl"
    sequences = [
        {
            "user_id": "u_001",
            "item_ids": ["i_001", "i_002", "i_003"],
            "event_weights": [1.0, 2.0, 3.0],
            "train_history_item_ids": ["i_001"],
            "validation_item_id": "i_002",
            "test_item_id": "i_003",
        }
    ]

    write_user_sequences(sequences, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == sequences
