import pandas as pd


def _reranker_tools():
    from cognid_genrec.retrieval.diversity_reranker import DiversityReranker
    from cognid_genrec.retrieval.post_filter import filter_seen_items

    return DiversityReranker, filter_seen_items


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": "i_old_tech",
                "topic": "technology",
                "author_id": "a_001",
                "publish_time": "2026-05-01T10:00:00",
                "quality_score": 0.90,
                "exposure_count": 80,
            },
            {
                "item_id": "i_new_tech",
                "topic": "technology",
                "author_id": "a_002",
                "publish_time": "2026-06-28T10:00:00",
                "quality_score": 0.82,
                "exposure_count": 30,
            },
            {
                "item_id": "i_finance",
                "topic": "finance",
                "author_id": "a_001",
                "publish_time": "2026-06-20T10:00:00",
                "quality_score": 0.78,
                "exposure_count": 40,
            },
            {
                "item_id": "i_tail_life",
                "topic": "lifestyle",
                "author_id": "a_003",
                "publish_time": "2026-05-20T10:00:00",
                "quality_score": 0.96,
                "exposure_count": 0,
            },
        ]
    )


def test_filter_seen_items_removes_history_from_candidates():
    _, filter_seen_items = _reranker_tools()
    candidates = [
        {"item_id": "i_seen", "score": 0.9},
        {"item_id": "i_new", "score": 0.8},
    ]

    assert filter_seen_items(candidates, {"i_seen"}) == [{"item_id": "i_new", "score": 0.8}]


def test_diversity_reranker_applies_freshness_author_topic_and_long_tail_rules():
    DiversityReranker, _ = _reranker_tools()
    candidates = [
        {"item_id": "i_old_tech", "score": 0.90},
        {"item_id": "i_new_tech", "score": 0.85},
        {"item_id": "i_finance", "score": 0.80},
        {"item_id": "i_tail_life", "score": 0.20},
    ]
    reranker = DiversityReranker(
        freshness_window_days=7,
        freshness_boost=0.20,
        max_per_author=1,
        long_tail_slots=1,
        long_tail_exposure_threshold=5,
        long_tail_quality_threshold=0.90,
        now="2026-06-29T10:00:00",
    )

    reranked = reranker.rerank(candidates, _items(), top_k=3)
    item_ids = [candidate["item_id"] for candidate in reranked]

    assert item_ids == ["i_new_tech", "i_finance", "i_tail_life"]
    assert len({candidate["author_id"] for candidate in reranked}) == 3
    assert [candidate["topic"] for candidate in reranked] == [
        "technology",
        "finance",
        "lifestyle",
    ]
    assert reranked[-1]["source_reason"] == "long_tail_protection"
