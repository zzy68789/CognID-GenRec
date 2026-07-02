from pathlib import Path

import pandas as pd
import pytest

SAMPLE_DIR = Path("data/samples")


def _validators():
    from cognid_genrec.data.validate import (
        validate_interactions,
        validate_items,
        validate_users,
    )

    return validate_items, validate_users, validate_interactions


def test_sample_schema_files_validate():
    validate_items, validate_users, validate_interactions = _validators()

    items = validate_items(SAMPLE_DIR / "items.csv")
    users = validate_users(SAMPLE_DIR / "users.csv")
    interactions = validate_interactions(SAMPLE_DIR / "interactions.csv")

    assert set(interactions["item_id"]).issubset(set(items["item_id"]))
    assert set(interactions["user_id"]).issubset(set(users["user_id"]))


def test_validate_items_rejects_missing_columns(tmp_path):
    validate_items, _, _ = _validators()

    path = tmp_path / "items_missing_body.csv"
    pd.DataFrame(
        [
            {
                "item_id": "i_001",
                "title": "A title",
                "topic": "tech",
                "author_id": "a_001",
                "publish_time": "2026-06-01T10:00:00",
                "quality_score": 0.8,
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="items missing columns"):
        validate_items(path)


def test_validate_items_rejects_duplicate_item_id(tmp_path):
    validate_items, _, _ = _validators()

    path = tmp_path / "items_duplicate.csv"
    pd.DataFrame(
        [
            {
                "item_id": "i_001",
                "title": "A title",
                "body": "Summary",
                "topic": "tech",
                "author_id": "a_001",
                "publish_time": "2026-06-01T10:00:00",
                "quality_score": 0.8,
            },
            {
                "item_id": "i_001",
                "title": "Another title",
                "body": "Another summary",
                "topic": "finance",
                "author_id": "a_002",
                "publish_time": "2026-06-01T11:00:00",
                "quality_score": 0.7,
            },
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicated item_id"):
        validate_items(path)


def test_validate_interactions_rejects_invalid_event_type(tmp_path):
    _, _, validate_interactions = _validators()

    path = tmp_path / "interactions_invalid_event.csv"
    pd.DataFrame(
        [
            {
                "user_id": "u_001",
                "item_id": "i_001",
                "event_type": "share",
                "event_time": "2026-06-01T12:00:00",
                "dwell_time": 18.0,
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="invalid event_type"):
        validate_interactions(path)
