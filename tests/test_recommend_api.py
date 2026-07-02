from fastapi.testclient import TestClient


def test_recommend_api_returns_ranked_items_with_explanations():
    from cognid_genrec.service.api import create_app

    client = TestClient(create_app(data_dir="data/processed"))

    response = client.post(
        "/recommend",
        json={
            "user_id": "u_001",
            "history_item_ids": ["i_001"],
            "top_k": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "u_001"
    assert 1 <= len(payload["items"]) <= 2
    assert "i_001" not in {item["item_id"] for item in payload["items"]}
    first_item = payload["items"][0]
    assert set(first_item) == {
        "item_id",
        "score",
        "semantic_id",
        "reason",
        "source",
        "recall_path",
        "rerank_reason",
    }
    assert isinstance(first_item["semantic_id"], list)
    assert first_item["recall_path"]
    assert any(step.startswith("generated_semantic_ids=") for step in first_item["recall_path"])
    assert first_item["rerank_reason"]
    assert first_item["source"] == "generative_rerank"
