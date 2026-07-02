import pandas as pd
import pytest


def _semantic_id_tools():
    from cognid_genrec.tokenizer.semantic_id_tokenizer import (
        RQVAESemanticIDTokenizer,
        SemanticIDTokenizer,
        collision_rate,
        topic_purity,
        write_semantic_id_quality_report,
    )

    return (
        RQVAESemanticIDTokenizer,
        SemanticIDTokenizer,
        collision_rate,
        topic_purity,
        write_semantic_id_quality_report,
    )


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": "i_001",
                "title": "AI funding",
                "body": "Startup funding product launch artificial intelligence",
                "topic": "technology",
            },
            {
                "item_id": "i_002",
                "title": "Recommendation metrics",
                "body": "Retrieval ranking recall ndcg recommender systems",
                "topic": "technology",
            },
            {
                "item_id": "i_003",
                "title": "Campus meals",
                "body": "Affordable student food and campus lifestyle guide",
                "topic": "lifestyle",
            },
            {
                "item_id": "i_004",
                "title": "Market risk",
                "body": "Daily market factors risk signals portfolio",
                "topic": "finance",
            },
        ]
    )


def test_semantic_id_tokenizer_outputs_multilevel_item_mapping():
    _, SemanticIDTokenizer, _, _, _ = _semantic_id_tools()

    mapping = SemanticIDTokenizer(
        num_levels=3,
        clusters_per_level=(2, 3, 4),
        random_state=7,
    ).fit_transform(_items())

    assert mapping["item_id"].tolist() == ["i_001", "i_002", "i_003", "i_004"]
    assert mapping["semantic_id"].map(len).eq(3).all()
    assert mapping["semantic_id"].map(lambda values: all(isinstance(v, int) for v in values)).all()
    assert mapping["semantic_id_str"].str.count("-").eq(2).all()


def test_rqvae_semantic_id_tokenizer_outputs_residual_code_path():
    RQVAESemanticIDTokenizer, _, collision_rate, topic_purity, _ = _semantic_id_tools()

    mapping = RQVAESemanticIDTokenizer(
        num_levels=3,
        codebook_size=3,
        embedding_dim=8,
        max_iter=3,
        random_state=13,
    ).fit_transform(_items())

    assert mapping["item_id"].tolist() == ["i_001", "i_002", "i_003", "i_004"]
    assert mapping["tokenizer_method"].eq("rqvae").all()
    assert mapping["semantic_id"].map(len).eq(3).all()
    assert mapping["semantic_id"].map(lambda values: all(isinstance(v, int) for v in values)).all()
    assert mapping["semantic_id"].map(lambda values: all(0 <= v < 3 for v in values)).all()
    assert mapping["semantic_id_str"].str.count("-").eq(2).all()
    assert 0.0 <= collision_rate(mapping) <= 1.0
    assert 0.0 <= topic_purity(mapping) <= 1.0


def test_rqvae_tokenizer_allows_codebook_larger_than_item_count():
    RQVAESemanticIDTokenizer, _, _, _, _ = _semantic_id_tools()

    mapping = RQVAESemanticIDTokenizer(
        num_levels=2,
        codebook_size=16,
        embedding_dim=8,
        max_iter=2,
        random_state=17,
    ).fit_transform(_items())

    assert len(mapping) == len(_items())
    assert mapping["semantic_id"].map(len).eq(2).all()


def test_train_script_selects_rqvae_tokenizer():
    from cognid_genrec.tokenizer.semantic_id_tokenizer import RQVAESemanticIDTokenizer
    from scripts.train_semantic_id import build_tokenizer

    assert isinstance(build_tokenizer("rqvae"), RQVAESemanticIDTokenizer)


def test_collision_rate_and_topic_purity_are_explicit_quality_metrics():
    _, _, collision_rate, topic_purity, _ = _semantic_id_tools()
    mapping = pd.DataFrame(
        [
            {"item_id": "i_001", "semantic_id_str": "0-0-0", "topic": "technology"},
            {"item_id": "i_002", "semantic_id_str": "0-0-0", "topic": "technology"},
            {"item_id": "i_003", "semantic_id_str": "1-0-0", "topic": "finance"},
        ]
    )

    assert collision_rate(mapping) == pytest.approx(1 / 3)
    assert topic_purity(mapping) == pytest.approx(1.0)


def test_semantic_id_quality_report_contains_metrics(tmp_path):
    _, _, _, _, write_semantic_id_quality_report = _semantic_id_tools()
    output_path = tmp_path / "semantic_id_quality.md"
    metrics = {"collision_rate": 0.25, "topic_purity": 0.75, "item_count": 4}

    write_semantic_id_quality_report(metrics, output_path)

    payload = output_path.read_text(encoding="utf-8")
    assert "# Semantic ID Quality" in payload
    assert "| collision_rate | 0.2500 |" in payload
    assert "| topic_purity | 0.7500 |" in payload
