import pandas as pd


def test_build_item_features_outputs_content_schema():
    from cognid_genrec.kuairec.features import build_item_features

    categories = pd.DataFrame({"video_id": [1], "feat": ["food|life"]})
    captions = pd.DataFrame(
        {"video_id": [1], "caption": ["home cooking"], "category": ["food"]}
    )
    daily = pd.DataFrame(
        {"video_id": [1], "show_cnt": [10], "play_cnt": [8], "like_cnt": [2]}
    )

    result = build_item_features(categories, captions, daily)

    assert result.loc[0, "item_id"] == "1"
    assert "home cooking" in result.loc[0, "body"]
    assert result.loc[0, "topic"] == "food"
    assert 0.0 <= result.loc[0, "quality_score"] <= 1.0
    assert result.loc[0, "hot_score"] > 0.0


def test_build_user_features_preserves_user_id_and_activity_bucket():
    from cognid_genrec.kuairec.features import build_user_features

    users = pd.DataFrame({"user_id": [1], "user_active_degree": ["high_active"]})
    interactions = pd.DataFrame({"user_id": [1, 1, 2], "video_id": [10, 11, 12]})

    result = build_user_features(users, interactions)

    assert set(result["user_id"]) == {"1", "2"}
    assert result.loc[result["user_id"] == "1", "activity_bucket"].item() == "medium"
    assert result.loc[result["user_id"] == "2", "activity_bucket"].item() == "short"
