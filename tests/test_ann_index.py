import numpy as np


def test_numpy_ann_index_returns_nearest_items_by_cosine_similarity():
    from cognid_genrec.retrieval.ann_index import NumpyANNIndex

    index = NumpyANNIndex(
        item_ids=["10", "20", "30"],
        embeddings=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        ),
    )

    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)

    assert [item_id for item_id, _ in results] == ["10", "30"]
    assert results[0][1] >= results[1][1]
