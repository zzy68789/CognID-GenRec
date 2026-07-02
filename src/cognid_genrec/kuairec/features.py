from __future__ import annotations

import math

import numpy as np
import pandas as pd


ITEM_FEATURE_COLUMNS = [
    "item_id",
    "title",
    "body",
    "topic",
    "author_id",
    "duration",
    "quality_score",
    "hot_score",
    "caption_text",
    "category_text",
]


def build_item_features(
    categories: pd.DataFrame | None = None,
    captions: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    categories = _normalize_video_frame(categories)
    captions = _normalize_video_frame(captions)
    daily = _normalize_video_frame(daily)

    video_ids = sorted(
        set(categories.get("video_id", pd.Series(dtype=str)).astype(str))
        | set(captions.get("video_id", pd.Series(dtype=str)).astype(str))
        | set(daily.get("video_id", pd.Series(dtype=str)).astype(str)),
        key=_stable_sort_key,
    )
    if not video_ids:
        return pd.DataFrame(columns=ITEM_FEATURE_COLUMNS)

    base = pd.DataFrame({"video_id": video_ids})
    merged = base.merge(_category_text(categories), on="video_id", how="left")
    merged = merged.merge(_caption_text(captions), on="video_id", how="left")
    merged = merged.merge(_daily_stats(daily), on="video_id", how="left")
    merged = merged.fillna(
        {
            "category_text": "",
            "caption_text": "",
            "topic": "",
            "author_id": "",
            "duration": 0.0,
            "quality_score": 0.0,
            "hot_score": 0.0,
        }
    )
    merged["topic"] = [
        topic or _first_token(category)
        for topic, category in zip(
            merged["topic"].astype(str),
            merged["category_text"].astype(str),
            strict=False,
        )
    ]
    merged["topic"] = merged["topic"].replace({"": "unknown"})
    merged["title"] = merged.apply(
        lambda row: row["caption_text"] or row["category_text"] or f"video {row['video_id']}",
        axis=1,
    )
    merged["body"] = merged.apply(
        lambda row: " ".join(
            part
            for part in [str(row["caption_text"]), str(row["category_text"])]
            if part and part != "nan"
        ),
        axis=1,
    )
    merged["item_id"] = merged["video_id"].astype(str)
    return merged[ITEM_FEATURE_COLUMNS].reset_index(drop=True)


def build_user_features(users: pd.DataFrame | None, interactions: pd.DataFrame) -> pd.DataFrame:
    users = users.copy() if users is not None else pd.DataFrame(columns=["user_id"])
    if "user_id" not in users.columns:
        users["user_id"] = pd.Series(dtype=str)
    user_ids = set(users["user_id"].astype(str))
    if "user_id" in interactions.columns:
        user_ids |= set(interactions["user_id"].astype(str))
    base = pd.DataFrame({"user_id": sorted(user_ids, key=_stable_sort_key)})
    counts = (
        interactions.assign(user_id=interactions["user_id"].astype(str))
        .groupby("user_id")
        .size()
        .rename("interaction_count")
        .reset_index()
        if "user_id" in interactions.columns and not interactions.empty
        else pd.DataFrame(columns=["user_id", "interaction_count"])
    )
    result = base.merge(counts, on="user_id", how="left")
    result["interaction_count"] = result["interaction_count"].fillna(0).astype(int)
    result["activity_bucket"] = result["interaction_count"].map(_activity_bucket)
    extra = users.assign(user_id=users["user_id"].astype(str)).drop_duplicates("user_id")
    extra_columns = [column for column in extra.columns if column != "user_id"]
    if extra_columns:
        result = result.merge(extra[["user_id", *extra_columns]], on="user_id", how="left")
    return result


def _normalize_video_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=["video_id"])
    result = frame.copy()
    if "item_id" in result.columns and "video_id" not in result.columns:
        result = result.rename(columns={"item_id": "video_id"})
    if "video_id" not in result.columns:
        result["video_id"] = pd.Series(dtype=str)
    result["video_id"] = result["video_id"].astype(str)
    return result


def _category_text(categories: pd.DataFrame) -> pd.DataFrame:
    if categories.empty:
        return pd.DataFrame(columns=["video_id", "category_text"])
    value_columns = [column for column in categories.columns if column != "video_id"]
    result = categories.copy()
    result["category_text"] = result[value_columns].apply(_join_values, axis=1) if value_columns else ""
    return result[["video_id", "category_text"]].groupby("video_id", as_index=False).agg(
        {"category_text": _merge_text_values}
    )


def _caption_text(captions: pd.DataFrame) -> pd.DataFrame:
    if captions.empty:
        return pd.DataFrame(columns=["video_id", "caption_text", "topic"])
    caption_columns = [
        column
        for column in ["caption", "caption_text", "title", "text", "ocr"]
        if column in captions.columns
    ]
    topic_columns = [
        column
        for column in ["category", "category_name", "first_level_category", "second_level_category", "topic"]
        if column in captions.columns
    ]
    result = captions.copy()
    result["caption_text"] = result[caption_columns].apply(_join_values, axis=1) if caption_columns else ""
    result["topic"] = result[topic_columns].apply(_first_non_empty, axis=1) if topic_columns else ""
    return result[["video_id", "caption_text", "topic"]].groupby("video_id", as_index=False).agg(
        {"caption_text": _merge_text_values, "topic": _first_text_value}
    )


def _daily_stats(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(
            columns=["video_id", "author_id", "duration", "quality_score", "hot_score"]
        )
    result = daily.copy()
    numeric_columns = [
        column
        for column in result.columns
        if column != "video_id" and pd.api.types.is_numeric_dtype(result[column])
    ]
    grouped = (
        result.groupby("video_id", as_index=False)[numeric_columns].mean()
        if numeric_columns
        else result[["video_id"]].drop_duplicates()
    )
    author_by_video = (
        result.groupby("video_id")["author_id"].first().astype(str)
        if "author_id" in result.columns
        else pd.Series(dtype=str)
    )
    grouped["author_id"] = grouped["video_id"].map(author_by_video).fillna("")
    grouped["duration"] = _numeric_or_default(grouped, "video_duration")
    show = _numeric_or_default(grouped, "show_cnt")
    play = _numeric_or_default(grouped, "play_cnt")
    like = _numeric_or_default(grouped, "like_cnt")
    comment = _numeric_or_default(grouped, "comment_cnt")
    share = _numeric_or_default(grouped, "share_cnt")
    denominator = np.maximum(play.to_numpy(dtype=float), show.to_numpy(dtype=float))
    denominator = np.maximum(denominator, 1.0)
    grouped["quality_score"] = np.clip(
        (like.to_numpy(dtype=float) + comment.to_numpy(dtype=float) + share.to_numpy(dtype=float))
        / denominator,
        0.0,
        1.0,
    )
    raw_hot = show.to_numpy(dtype=float) + play.to_numpy(dtype=float) + like.to_numpy(dtype=float)
    max_hot = float(np.max(np.log1p(raw_hot))) if len(raw_hot) else 0.0
    grouped["hot_score"] = 0.0 if max_hot <= 0 else np.log1p(raw_hot) / max_hot
    return grouped[["video_id", "author_id", "duration", "quality_score", "hot_score"]]


def _numeric_or_default(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([0.0] * len(frame), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _join_values(row: pd.Series) -> str:
    return " ".join(str(value) for value in row.tolist() if _has_text(value))


def _merge_text_values(values: pd.Series) -> str:
    return " ".join(str(value) for value in values.tolist() if _has_text(value))


def _first_non_empty(row: pd.Series) -> str:
    for value in row.tolist():
        if _has_text(value):
            return str(value)
    return ""


def _first_text_value(values: pd.Series) -> str:
    for value in values.tolist():
        if _has_text(value):
            return str(value)
    return ""


def _first_token(value: str) -> str:
    if not value:
        return ""
    return value.replace("|", " ").split()[0]


def _has_text(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def _activity_bucket(count: int) -> str:
    if count < 2:
        return "short"
    if count < 10:
        return "medium"
    return "long"


def _stable_sort_key(value: object) -> tuple[int, object]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)
