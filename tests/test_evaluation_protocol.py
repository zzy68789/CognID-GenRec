import json

import numpy as np
import pytest


def _sequences():
    return [
        {
            "user_id": "u1",
            "item_ids": ["1", "2", "3", "4"],
            "actions": ["a1", "a2", "a3", "a4"],
            "action_weights": [1.0, 1.0, 1.5, 2.0],
            "event_weights": [1.0, 1.0, 1.5, 2.0],
            "timestamps": [10, 20, 30, 40],
            "time_deltas": [0, 10, 10, 10],
            "train_history_item_ids": ["1", "2"],
            "validation_history_item_ids": ["1", "2"],
            "test_history_item_ids": ["1", "2", "3"],
            "validation_item_id": "3",
            "test_item_id": "4",
            "test_action": "a4",
        },
        {
            "user_id": "u2",
            "item_ids": ["2", "5", "6", "1"],
            "actions": ["a1", "a2", "a3", "a4"],
            "action_weights": [1.0, 1.0, 1.5, 2.0],
            "event_weights": [1.0, 1.0, 1.5, 2.0],
            "timestamps": [10, 20, 30, 40],
            "time_deltas": [0, 10, 10, 10],
            "train_history_item_ids": ["2", "5"],
            "validation_history_item_ids": ["2", "5"],
            "test_history_item_ids": ["2", "5", "6"],
            "validation_item_id": "6",
            "test_item_id": "1",
            "test_action": "a4",
        },
    ]


def _manifest():
    from cognid_genrec.evaluation.protocol import build_sequential_candidate_manifest

    return build_sequential_candidate_manifest(
        sequences=_sequences(),
        candidate_item_ids=["1", "2", "3", "4", "5", "6"],
        dataset_matrix="big",
        random_seed=7,
    )


def test_candidate_manifest_round_trip_and_train_only_boundary(tmp_path):
    from cognid_genrec.evaluation.protocol import (
        build_train_only_sequences,
        read_candidate_manifest,
        write_candidate_manifest,
    )

    path = tmp_path / "candidate_manifest.json"
    write_candidate_manifest(_manifest(), path)
    loaded = read_candidate_manifest(path)
    train_rows = build_train_only_sequences(_sequences())

    assert loaded.candidate_item_ids == ("1", "2", "3", "4", "5", "6")
    assert loaded.random_seed == 7
    assert json.loads(path.read_text(encoding="utf-8"))["candidate_count"] == 6
    assert train_rows[0]["item_ids"] == ["1", "2"]
    assert train_rows[0]["actions"] == ["a1", "a2"]
    assert train_rows[0]["timestamps"] == [10, 20]
    assert train_rows[0]["protocol_view"] == "train_only"


def test_protocol_rejects_unreachable_target_and_split_leakage():
    from cognid_genrec.evaluation.protocol import (
        CandidateManifest,
        validate_sequential_protocol,
    )

    unreachable = CandidateManifest(
        protocol_name="sequential_next_item",
        dataset_matrix="big",
        candidate_source="test",
        candidate_item_ids=("1", "2", "3"),
    )
    with pytest.raises(ValueError, match="outside the candidate universe"):
        validate_sequential_protocol(_sequences(), unreachable)

    leaked = _sequences()
    leaked[0]["train_history_item_ids"] = ["1", "2", "3"]
    with pytest.raises(ValueError, match="crosses the held-out boundary"):
        validate_sequential_protocol(leaked, _manifest())


def test_protocol_rejects_recommendations_outside_candidates():
    from cognid_genrec.evaluation.protocol import validate_recommendations

    with pytest.raises(ValueError, match="outside the candidate universe"):
        validate_recommendations(
            recommendations={"u1": ["outside"], "u2": ["1"]},
            sequences=_sequences(),
            manifest=_manifest(),
            top_k=1,
        )


def test_all_retrievers_share_manifest_and_exclude_test_history(tmp_path):
    from scripts.evaluate_kuairec_retriever import recommend

    item_ids = ["1", "2", "3", "4", "5", "6"]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    np.save(tmp_path / "ann_items.npy", embeddings)
    (tmp_path / "ann_item_ids.json").write_text(json.dumps(item_ids), encoding="utf-8")

    manifest = _manifest()
    candidates = manifest.candidate_set
    histories = {
        row["user_id"]: set(row["test_history_item_ids"]) for row in _sequences()
    }
    for method in ["popular", "itemcf", "ann_transformer"]:
        recommendations = recommend(method, tmp_path, _sequences(), manifest, top_k=2)
        for user_id, ranked in recommendations.items():
            assert len(ranked) == 2
            assert set(ranked) <= candidates
            assert not (set(ranked) & histories[user_id])
