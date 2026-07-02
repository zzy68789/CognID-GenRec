from __future__ import annotations

INTERACTION_COLUMNS = {"user_id", "video_id", "timestamp", "watch_ratio"}
ITEM_CATEGORY_COLUMNS = {"video_id"}
CAPTION_CATEGORY_COLUMNS = {"video_id"}
USER_FEATURE_COLUMNS = {"user_id"}
ITEM_DAILY_COLUMNS = {"video_id"}

RAW_FILES = {
    "small": "small_matrix.csv",
    "big": "big_matrix.csv",
    "social_network": "social_network.csv",
    "user_features": "user_features.csv",
    "user_features_raw": "user_features_raw.csv",
    "item_daily_features": "item_daily_features.csv",
    "item_categories": "item_categories.csv",
    "caption_category": "kuairec_caption_category.csv",
}

ACTION_LABELS = (
    "short_view",
    "valid_view",
    "complete_view",
    "high_interest",
)
