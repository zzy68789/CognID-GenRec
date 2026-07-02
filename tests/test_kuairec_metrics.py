import pandas as pd
import pytest


def test_hit_rate_and_weighted_hit_rate_at_k():
    from cognid_genrec.evaluation.segmented_metrics import (
        hit_rate_at_k,
        weighted_hit_rate_at_k,
    )

    assert hit_rate_at_k(["1", "2", "3"], target="2", k=2) == 1.0
    assert hit_rate_at_k(["1", "2", "3"], target="3", k=2) == 0.0
    assert weighted_hit_rate_at_k(["1", "2"], target="2", target_weight=1.5, k=2) == 1.5


def test_segmented_kuairec_report_contains_core_metrics_and_slices(tmp_path):
    from cognid_genrec.evaluation.segmented_metrics import (
        evaluate_segmented_recommendations,
        write_segmented_metrics_report,
    )

    sequences = [
        {
            "user_id": "u1",
            "item_ids": ["1", "2", "3", "4"],
            "train_history_item_ids": ["1", "2"],
            "test_item_id": "4",
            "test_action": "high_interest",
            "test_action_weight": 2.5,
        },
        {
            "user_id": "u2",
            "item_ids": ["2", "3", "5"],
            "train_history_item_ids": ["2"],
            "test_item_id": "5",
            "test_action": "complete_view",
            "test_action_weight": 1.5,
        },
    ]
    recommendations = {"u1": ["4", "7"], "u2": ["8", "5"]}
    item_features = pd.DataFrame(
        {
            "item_id": ["4", "5", "7", "8"],
            "topic": ["food", "music", "food", "sports"],
            "hot_score": [0.9, 0.1, 0.4, 0.2],
        }
    )

    report = evaluate_segmented_recommendations(
        recommendations=recommendations,
        sequences=sequences,
        item_features=item_features,
        k_values=(10, 20),
    )

    assert report["all"]["HR@10"] == 1.0
    assert report["all"]["NDCG@10"] == pytest.approx((1.0 + 1.0 / 1.5849625007) / 2.0)
    assert report["all"]["Recall@20"] == 1.0
    assert report["all"]["Coverage"] > 0.0
    assert report["all"]["Diversity"] > 0.0
    assert "action=high_interest" in report
    assert "activity=short" in report
    assert any(key.startswith("popularity=") for key in report)

    output_path = tmp_path / "metrics.md"
    write_segmented_metrics_report("popular", report, output_path)
    payload = output_path.read_text(encoding="utf-8")
    assert "HR@10" in payload
    assert "action=high_interest" in payload
