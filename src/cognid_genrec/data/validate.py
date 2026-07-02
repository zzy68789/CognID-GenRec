from pathlib import Path

import pandas as pd

from cognid_genrec.data.schemas import (
    INTERACTION_COLUMNS,
    ITEM_COLUMNS,
    USER_COLUMNS,
    VALID_EVENT_TYPES,
)


def validate_items(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, ITEM_COLUMNS, "items")
    _require_unique(df, "item_id", "items")
    df = df.copy()
    df["publish_time"] = _parse_datetime(df, "publish_time", "items")
    df["quality_score"] = _parse_numeric(df, "quality_score", "items")
    return df


def validate_users(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, USER_COLUMNS, "users")
    _require_unique(df, "user_id", "users")
    return df


def validate_interactions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, INTERACTION_COLUMNS, "interactions")
    invalid_events = sorted(set(df["event_type"].dropna()) - VALID_EVENT_TYPES)
    if invalid_events:
        raise ValueError(f"invalid event_type: {invalid_events}")
    df = df.copy()
    df["event_time"] = _parse_datetime(df, "event_time", "interactions")
    df["dwell_time"] = _parse_numeric(df, "dwell_time", "interactions")
    if (df["dwell_time"] < 0).any():
        raise ValueError("interactions contains negative dwell_time")
    return df


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _require_unique(df: pd.DataFrame, column: str, label: str) -> None:
    if df[column].duplicated().any():
        raise ValueError(f"{label} contains duplicated {column}")


def _parse_datetime(df: pd.DataFrame, column: str, label: str) -> pd.Series:
    try:
        return pd.to_datetime(df[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains invalid {column}") from exc


def _parse_numeric(df: pd.DataFrame, column: str, label: str) -> pd.Series:
    try:
        return pd.to_numeric(df[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains invalid {column}") from exc
