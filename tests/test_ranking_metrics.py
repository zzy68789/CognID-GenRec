import json

import pandas as pd
import pytest


def _metrics():
    from cognid_genrec.evaluation.ranking_metrics import (
        coverage_at_k,
        ndcg_at_k,
        recall_at_k,
        topic_diversity_at_k,
    )

    return recall_at_k, ndcg_at_k, coverage_at_k, topic_diversity_at_k


def _baselines():
    from cognid_genrec.models.baselines.itemcf import ItemCFRecommender
    from cognid_genrec.models.baselines.popular import PopularRecommender

    return PopularRecommender, ItemCFRecommender


def test_recall_ndcg_coverage_and_diversity_metrics():
    recall_at_k, ndcg_at_k, coverage_at_k, topic_diversity_at_k = _metrics()
    recommendations = {
        "u_001": ["i_002", "i_003", "i_001"],
        "u_002": ["i_004", "i_005"],
    }
    targets = {"u_001": "i_001", "u_002": "i_009"}
    item_topics = {
        "i_001": "technology",
        "i_002": "technology",
        "i_003": "technology",
        "i_004": "finance",
        "i_005": "lifestyle",
    }

    assert recall_at_k(recommendations, targets, 3) == 0.5
    assert ndcg_at_k(recommendations, targets, 3) == pytest.approx(0.25)
    assert coverage_at_k(recommendations, set(item_topics), 2) == pytest.approx(0.8)
    assert topic_diversity_at_k(recommendations, item_topics, 2) == pytest.approx(0.75)


def test_popular_recommender_ranks_weighted_positive_events_and_excludes_history():
    PopularRecommender, _ = _baselines()
    interactions = pd.DataFrame(
        [
            {"item_id": "i_001", "event_type": "click"},
            {"item_id": "i_002", "event_type": "like"},
            {"item_id": "i_003", "event_type": "collect"},
            {"item_id": "i_004", "event_type": "skip"},
        ]
    )

    recommender = PopularRecommender().fit(interactions)

    assert recommender.recommend(history_item_ids=["i_003"], top_k=3) == ["i_002", "i_001"]


def test_itemcf_recommender_uses_user_cooccurrence_and_excludes_history():
    _, ItemCFRecommender = _baselines()
    sequences = [
        {"user_id": "u_001", "item_ids": ["i_001", "i_002", "i_003"]},
        {"user_id": "u_002", "item_ids": ["i_001", "i_002", "i_004"]},
    ]

    recommender = ItemCFRecommender().fit(sequences)

    assert recommender.recommend(history_item_ids=["i_001"], top_k=3) == [
        "i_002",
        "i_003",
        "i_004",
    ]


def test_itemcf_recommender_caps_long_histories_and_neighbors():
    _, ItemCFRecommender = _baselines()
    sequences = [
        {"user_id": "u_001", "item_ids": ["i_001", "i_002", "i_003", "i_004"]},
    ]

    recommender = ItemCFRecommender(max_items_per_user=3, max_neighbors_per_item=1).fit(
        sequences
    )

    assert "i_001" not in recommender.item_similarities
    assert all(len(neighbors) <= 1 for neighbors in recommender.item_similarities.values())
    assert recommender.recommend(history_item_ids=["i_002"], top_k=2) == ["i_003"]


def test_evaluate_retrieval_writes_markdown_report(tmp_path):
    from cognid_genrec.evaluation.ranking_metrics import evaluate_recommendations
    from cognid_genrec.evaluation.reports import write_metrics_report

    recommendations = {"u_001": ["i_005", "i_002", "i_003"]}
    targets = {"u_001": "i_005"}
    item_topics = {"i_002": "technology", "i_003": "lifestyle", "i_005": "engineering"}
    metrics = evaluate_recommendations(
        recommendations=recommendations,
        targets=targets,
        all_item_ids=set(item_topics),
        item_topics=item_topics,
        k_values=(5, 10),
    )
    output_path = tmp_path / "popular.md"

    write_metrics_report("popular", metrics, output_path)

    payload = output_path.read_text(encoding="utf-8")
    assert "# Retrieval Evaluation: popular" in payload
    assert "| Recall@5 | 1.0000 |" in payload
    assert "| NDCG@10 | 1.0000 |" in payload
    assert json.dumps(metrics, sort_keys=True) in payload
