from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


SEQUENTIAL_NEXT_ITEM = "sequential_next_item"


@dataclass(frozen=True)
class CandidateManifest:
    protocol_name: str
    dataset_matrix: str
    candidate_source: str
    candidate_item_ids: tuple[str, ...]
    history_field: str = "test_history_item_ids"
    target_field: str = "test_item_id"
    fit_history_field: str = "train_history_item_ids"
    ranking_mode: str = "full_sort"
    random_seed: int = 42
    schema_version: int = 1

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_item_ids)

    @property
    def candidate_set(self) -> set[str]:
        return set(self.candidate_item_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_name": self.protocol_name,
            "dataset_matrix": self.dataset_matrix,
            "candidate_source": self.candidate_source,
            "candidate_count": self.candidate_count,
            "candidate_item_ids": list(self.candidate_item_ids),
            "history_field": self.history_field,
            "target_field": self.target_field,
            "fit_history_field": self.fit_history_field,
            "ranking_mode": self.ranking_mode,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CandidateManifest":
        candidate_item_ids = tuple(
            str(item_id) for item_id in payload["candidate_item_ids"]
        )
        expected_count = int(payload.get("candidate_count", len(candidate_item_ids)))
        if expected_count != len(candidate_item_ids):
            raise ValueError(
                "candidate manifest count does not match candidate_item_ids: "
                f"expected={expected_count}, actual={len(candidate_item_ids)}"
            )
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            protocol_name=str(payload["protocol_name"]),
            dataset_matrix=str(payload["dataset_matrix"]),
            candidate_source=str(payload["candidate_source"]),
            candidate_item_ids=candidate_item_ids,
            history_field=str(payload.get("history_field", "test_history_item_ids")),
            target_field=str(payload.get("target_field", "test_item_id")),
            fit_history_field=str(
                payload.get("fit_history_field", "train_history_item_ids")
            ),
            ranking_mode=str(payload.get("ranking_mode", "full_sort")),
            random_seed=int(payload.get("random_seed", 42)),
        )


def build_sequential_candidate_manifest(
    sequences: Sequence[dict],
    candidate_item_ids: Iterable[object],
    dataset_matrix: str,
    random_seed: int = 42,
) -> CandidateManifest:
    candidates = tuple(
        sorted({str(item_id) for item_id in candidate_item_ids}, key=_stable_item_key)
    )
    manifest = CandidateManifest(
        protocol_name=SEQUENTIAL_NEXT_ITEM,
        dataset_matrix=dataset_matrix,
        candidate_source="prepared_interactions",
        candidate_item_ids=candidates,
        random_seed=random_seed,
    )
    validate_sequential_protocol(sequences, manifest)
    return manifest


def write_candidate_manifest(
    manifest: CandidateManifest, output_path: str | Path
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_candidate_manifest(path: str | Path) -> CandidateManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CandidateManifest.from_dict(payload)


def validate_sequential_protocol(
    sequences: Sequence[dict],
    manifest: CandidateManifest,
) -> None:
    if manifest.schema_version != 1:
        raise ValueError(
            f"unsupported candidate manifest schema: {manifest.schema_version}"
        )
    if manifest.protocol_name != SEQUENTIAL_NEXT_ITEM:
        raise ValueError(f"unsupported evaluation protocol: {manifest.protocol_name}")
    if manifest.ranking_mode != "full_sort":
        raise ValueError(
            "sequential_next_item currently requires ranking_mode=full_sort"
        )
    if not manifest.candidate_item_ids:
        raise ValueError("candidate universe must not be empty")
    if len(manifest.candidate_set) != manifest.candidate_count:
        raise ValueError("candidate universe contains duplicate item IDs")
    expected_fields = {
        "fit_history_field": "train_history_item_ids",
        "history_field": "test_history_item_ids",
        "target_field": "test_item_id",
    }
    for field_name, expected_value in expected_fields.items():
        actual_value = getattr(manifest, field_name)
        if actual_value != expected_value:
            raise ValueError(
                f"sequential protocol requires {field_name}={expected_value}, "
                f"actual={actual_value}"
            )

    seen_users: set[str] = set()
    unreachable_targets: list[str] = []
    for sequence in sequences:
        user_id = str(sequence.get("user_id", ""))
        if not user_id:
            raise ValueError("evaluation sequence is missing user_id")
        if user_id in seen_users:
            raise ValueError(f"duplicate evaluation sequence for user_id={user_id}")
        seen_users.add(user_id)

        full_items = [str(item_id) for item_id in sequence.get("item_ids", [])]
        train_history = [
            str(item_id) for item_id in sequence.get(manifest.fit_history_field, [])
        ]
        test_history = [
            str(item_id) for item_id in sequence.get(manifest.history_field, [])
        ]
        validation_target = str(sequence.get("validation_item_id", ""))
        test_target = str(sequence.get(manifest.target_field, ""))
        if len(full_items) < 2 or not validation_target or not test_target:
            raise ValueError(f"user_id={user_id} has an incomplete chronological split")
        if train_history != full_items[:-2]:
            raise ValueError(
                f"user_id={user_id} train history crosses the held-out boundary"
            )
        if validation_target != full_items[-2]:
            raise ValueError(
                f"user_id={user_id} validation target is not the penultimate event"
            )
        if test_history != full_items[:-1]:
            raise ValueError(
                f"user_id={user_id} test history must include validation but exclude the test event"
            )
        if test_target != full_items[-1]:
            raise ValueError(f"user_id={user_id} test target is not the final event")
        if test_target not in manifest.candidate_set:
            unreachable_targets.append(f"{user_id}:{test_target}")

    if unreachable_targets:
        examples = ", ".join(unreachable_targets[:5])
        raise ValueError(
            f"{len(unreachable_targets)} test targets are outside the candidate universe; "
            f"examples={examples}"
        )


def build_train_only_sequences(sequences: Sequence[dict]) -> list[dict]:
    train_sequences: list[dict] = []
    aligned_fields = (
        "actions",
        "action_weights",
        "event_weights",
        "timestamps",
        "time_deltas",
    )
    for sequence in sequences:
        train_history = [
            str(item_id) for item_id in sequence.get("train_history_item_ids", [])
        ]
        row = dict(sequence)
        row["item_ids"] = train_history
        row["sequence_length"] = len(train_history)
        row["protocol_view"] = "train_only"
        for field in aligned_fields:
            if field not in sequence:
                continue
            values = list(sequence[field])
            if len(values) < len(train_history):
                raise ValueError(
                    f"user_id={sequence.get('user_id')} field={field} is shorter than train history"
                )
            row[field] = values[: len(train_history)]
        train_sequences.append(row)
    return train_sequences


def evaluation_histories(
    sequences: Sequence[dict],
    manifest: CandidateManifest,
) -> dict[str, list[str]]:
    return {
        str(sequence["user_id"]): [
            str(item_id) for item_id in sequence.get(manifest.history_field, [])
        ]
        for sequence in sequences
    }


def validate_recommendations(
    recommendations: Mapping[str, Sequence[str]],
    sequences: Sequence[dict],
    manifest: CandidateManifest,
    top_k: int,
) -> None:
    expected_users = {str(sequence["user_id"]) for sequence in sequences}
    actual_users = {str(user_id) for user_id in recommendations}
    if actual_users != expected_users:
        missing = sorted(expected_users - actual_users)
        extra = sorted(actual_users - expected_users)
        raise ValueError(
            f"recommendation users do not match evaluation users: missing={missing}, extra={extra}"
        )

    histories = evaluation_histories(sequences, manifest)
    for user_id in sorted(expected_users):
        ranked = [str(item_id) for item_id in recommendations[user_id]]
        if len(ranked) != len(set(ranked)):
            raise ValueError(
                f"recommendations contain duplicate items for user_id={user_id}"
            )
        outside = [
            item_id for item_id in ranked if item_id not in manifest.candidate_set
        ]
        if outside:
            raise ValueError(
                f"recommendations contain items outside the candidate universe for user_id={user_id}: "
                f"{outside[:5]}"
            )
        history = set(histories[user_id])
        leaked = [item_id for item_id in ranked if item_id in history]
        if leaked:
            raise ValueError(
                f"recommendations contain historical items for user_id={user_id}: {leaked[:5]}"
            )
        eligible_count = len(manifest.candidate_set - history)
        expected_count = min(top_k, eligible_count)
        if len(ranked) != expected_count:
            raise ValueError(
                f"recommendation count mismatch for user_id={user_id}: "
                f"expected={expected_count}, actual={len(ranked)}"
            )


def _stable_item_key(value: str) -> tuple[int, object]:
    return (0, int(value)) if value.isdigit() else (1, value)
