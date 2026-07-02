from __future__ import annotations

import pandas as pd

ACTION_WEIGHTS = {
    "short_view": 0.2,
    "valid_view": 1.0,
    "complete_view": 1.5,
    "high_interest": 2.5,
}

ACTION_TO_ID = {
    "pad": 0,
    "short_view": 1,
    "valid_view": 2,
    "complete_view": 3,
    "high_interest": 4,
}


def action_from_watch_ratio(watch_ratio: float) -> str:
    value = float(watch_ratio)
    if value < 0.3:
        return "short_view"
    if value < 1.0:
        return "valid_view"
    if value < 2.0:
        return "complete_view"
    return "high_interest"


def attach_actions(interactions: pd.DataFrame) -> pd.DataFrame:
    if "watch_ratio" not in interactions.columns:
        raise ValueError("interactions missing column: watch_ratio")
    result = interactions.copy()
    result["action"] = result["watch_ratio"].map(action_from_watch_ratio)
    result["action_weight"] = result["action"].map(ACTION_WEIGHTS).astype(float)
    return result
