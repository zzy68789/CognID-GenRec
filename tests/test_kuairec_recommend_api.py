from fastapi.testclient import TestClient


def test_kuairec_api_returns_ann_recommendations():
    from cognid_genrec.service.kuairec_api import create_kuairec_app

    client = TestClient(create_kuairec_app(data_dir="data/processed/kuairec_sample"))
    response = client.post(
        "/recommend",
        json={
            "user_id": "1",
            "history_item_ids": ["10", "20"],
            "history_actions": ["valid_view", "high_interest"],
            "top_k": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "1"
    assert payload["items"]
    assert {
        "item_id",
        "score",
        "semantic_id",
        "source",
        "action_context",
        "ann_rank",
        "reason",
    } <= set(payload["items"][0])
