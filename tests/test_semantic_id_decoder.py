import pandas as pd


def _retrieval_tools():
    from cognid_genrec.models.generative_retriever import GenerativeRetriever
    from cognid_genrec.retrieval.semantic_id_decoder import SemanticIDDecoder

    return SemanticIDDecoder, GenerativeRetriever


def _semantic_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item_id": "i_001", "semantic_id": [0, 0, 0], "semantic_id_str": "0-0-0"},
            {"item_id": "i_002", "semantic_id": [1, 0, 0], "semantic_id_str": "1-0-0"},
            {"item_id": "i_005", "semantic_id": [2, 0, 0], "semantic_id_str": "2-0-0"},
        ]
    )


def test_semantic_id_decoder_maps_generated_ids_back_to_items():
    SemanticIDDecoder, _ = _retrieval_tools()
    mapping = pd.DataFrame(
        [
            {"item_id": "i_001", "semantic_id_str": "0-0-0"},
            {"item_id": "i_002", "semantic_id_str": "0-0-0"},
            {"item_id": "i_003", "semantic_id_str": "1-0-0"},
        ]
    )

    decoder = SemanticIDDecoder.from_mapping(mapping)

    assert decoder.decode(["0-0-0", "1-0-0"], top_k=3) == ["i_001", "i_002", "i_003"]


def test_generative_retriever_uses_beam_search_and_decodes_items():
    _, GenerativeRetriever = _retrieval_tools()
    sequences = [
        {"user_id": "u_001", "item_ids": ["i_001", "i_002", "i_005"]},
        {"user_id": "u_002", "item_ids": ["i_001", "i_002"]},
    ]

    retriever = GenerativeRetriever(beam_width=2, max_steps=2).fit(
        user_sequences=sequences,
        semantic_id_mapping=_semantic_mapping(),
    )

    assert retriever.generate_semantic_ids(["i_001"]) == ["1-0-0", "2-0-0"]
    assert retriever.recommend(["i_001"], top_k=3) == ["i_002", "i_005"]


def test_generative_retriever_artifact_roundtrip(tmp_path):
    _, GenerativeRetriever = _retrieval_tools()
    sequences = [{"user_id": "u_001", "item_ids": ["i_001", "i_002", "i_005"]}]
    artifact_path = tmp_path / "generative_retriever.json"

    retriever = GenerativeRetriever(beam_width=2, max_steps=2).fit(
        user_sequences=sequences,
        semantic_id_mapping=_semantic_mapping(),
    )
    retriever.save(artifact_path)
    loaded = GenerativeRetriever.load(artifact_path)

    assert loaded.recommend(["i_001"], top_k=3) == ["i_002", "i_005"]
