from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import ParserError

from cognid_genrec.kuairec.schemas import (
    CAPTION_CATEGORY_COLUMNS,
    INTERACTION_COLUMNS,
    ITEM_CATEGORY_COLUMNS,
    ITEM_DAILY_COLUMNS,
    RAW_FILES,
    USER_FEATURE_COLUMNS,
)


def read_matrix(path: str | Path) -> pd.DataFrame:
    frame = read_csv(path)
    frame = normalize_matrix_columns(frame)
    missing = INTERACTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"matrix missing columns: {sorted(missing)}")
    return frame


def normalize_matrix_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rename_map = {}
    if "item_id" in result.columns and "video_id" not in result.columns:
        rename_map["item_id"] = "video_id"
    if "time" in result.columns and "timestamp" not in result.columns:
        rename_map["time"] = "timestamp"
    if "event_time" in result.columns and "timestamp" not in result.columns:
        rename_map["event_time"] = "timestamp"
    if rename_map:
        result = result.rename(columns=rename_map)
    if "watch_ratio" not in result.columns and {"play_duration", "video_duration"} <= set(result.columns):
        duration = pd.to_numeric(result["video_duration"], errors="coerce").clip(lower=1e-12)
        result["watch_ratio"] = pd.to_numeric(result["play_duration"], errors="coerce") / duration
    if "timestamp" not in result.columns:
        result["timestamp"] = pd.Series(range(len(result)), index=result.index)
    result["timestamp"] = pd.to_numeric(result["timestamp"], errors="coerce")
    for fallback_column in ["time", "event_time"]:
        if fallback_column in result.columns:
            fallback = pd.to_numeric(result[fallback_column], errors="coerce")
            result["timestamp"] = result["timestamp"].fillna(fallback)
    result["timestamp"] = result["timestamp"].fillna(pd.Series(range(len(result)), index=result.index))
    return result


def load_raw_kuairec(raw_dir: str | Path, matrix: str = "small") -> dict[str, pd.DataFrame]:
    root = Path(raw_dir)
    matrix_name = matrix.lower()
    if matrix_name not in {"small", "big"}:
        raise ValueError("matrix must be 'small' or 'big'")

    interactions = read_matrix(resolve_raw_file(root, RAW_FILES[matrix_name]))
    return {
        "interactions": interactions,
        "item_categories": read_optional_csv(
            resolve_raw_file(root, RAW_FILES["item_categories"], required=False),
            ITEM_CATEGORY_COLUMNS,
        ),
        "caption_category": read_optional_csv(
            resolve_raw_file(root, RAW_FILES["caption_category"], required=False),
            CAPTION_CATEGORY_COLUMNS,
        ),
        "item_daily_features": read_optional_csv(
            resolve_raw_file(root, RAW_FILES["item_daily_features"], required=False),
            ITEM_DAILY_COLUMNS,
        ),
        "user_features": read_optional_csv(
            resolve_raw_file(root, RAW_FILES["user_features"], required=False),
            USER_FEATURE_COLUMNS,
        ),
    }


def resolve_raw_file(raw_dir: str | Path, filename: str, required: bool = True) -> Path:
    root = Path(raw_dir)
    candidates = [
        root / filename,
        root / "data" / filename,
        root / "KuaiRec 2.0" / "data" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(root.rglob(filename)) if root.exists() else []
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f"KuaiRec raw file not found: {filename} under {root}")
    return root / filename


def read_optional_csv(path: str | Path, required_columns: set[str]) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame(columns=sorted(required_columns))
    frame = read_csv(csv_path)
    frame = normalize_video_id_column(frame)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {sorted(missing)}")
    return frame


def normalize_video_id_column(frame: pd.DataFrame) -> pd.DataFrame:
    if "item_id" in frame.columns and "video_id" not in frame.columns:
        return frame.rename(columns={"item_id": "video_id"})
    return frame


def read_csv(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except ParserError:
        return pd.read_csv(path, engine="python")
