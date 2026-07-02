import pandas as pd
import pytest


def test_watch_ratio_to_action_label():
    from cognid_genrec.kuairec.behaviors import action_from_watch_ratio

    assert action_from_watch_ratio(0.1) == "short_view"
    assert action_from_watch_ratio(0.7) == "valid_view"
    assert action_from_watch_ratio(1.2) == "complete_view"
    assert action_from_watch_ratio(2.4) == "high_interest"


def test_attach_actions_adds_label_and_weight():
    from cognid_genrec.kuairec.behaviors import attach_actions

    frame = pd.DataFrame({"watch_ratio": [0.2, 0.8, 1.4, 2.2]})
    result = attach_actions(frame)

    assert result["action"].tolist() == [
        "short_view",
        "valid_view",
        "complete_view",
        "high_interest",
    ]
    assert result["action_weight"].tolist() == [0.2, 1.0, 1.5, 2.5]


def test_read_matrix_requires_kuairec_interaction_columns(tmp_path):
    from cognid_genrec.kuairec.loaders import read_matrix

    good_path = tmp_path / "small_matrix.csv"
    pd.DataFrame(
        {
            "user_id": [1],
            "video_id": [10],
            "timestamp": [100],
            "watch_ratio": [0.8],
        }
    ).to_csv(good_path, index=False)

    assert read_matrix(good_path)["video_id"].tolist() == [10]

    bad_path = tmp_path / "bad_matrix.csv"
    pd.DataFrame({"user_id": [1], "video_id": [10]}).to_csv(bad_path, index=False)
    with pytest.raises(ValueError, match="matrix missing columns"):
        read_matrix(bad_path)
