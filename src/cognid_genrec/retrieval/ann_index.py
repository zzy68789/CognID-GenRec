from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NumpyANNIndex:
    def __init__(self, item_ids: list[str], embeddings: np.ndarray) -> None:
        if len(item_ids) != len(embeddings):
            raise ValueError("item_ids and embeddings must have the same length")
        self.item_ids = [str(item_id) for item_id in item_ids]
        vectors = embeddings.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
        self.embeddings = vectors / norms

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        vector = query.astype(np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        scores = self.embeddings @ vector
        order = np.argsort(-scores)[:top_k]
        return [(self.item_ids[index], float(scores[index])) for index in order]

    @classmethod
    def load(cls, embeddings_path: str | Path, item_ids_path: str | Path) -> "NumpyANNIndex":
        embeddings = np.load(embeddings_path)
        item_ids = json.loads(Path(item_ids_path).read_text(encoding="utf-8"))
        return cls(item_ids=item_ids, embeddings=embeddings)

    def save(self, embeddings_path: str | Path, item_ids_path: str | Path) -> None:
        embeddings_output = Path(embeddings_path)
        ids_output = Path(item_ids_path)
        embeddings_output.parent.mkdir(parents=True, exist_ok=True)
        ids_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_output, self.embeddings)
        ids_output.write_text(json.dumps(self.item_ids, ensure_ascii=False, indent=2), encoding="utf-8")


try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - faiss is optional.
    faiss = None


class FaissANNIndex:
    def __init__(self, item_ids: list[str], embeddings: np.ndarray) -> None:
        if faiss is None:
            raise ImportError("faiss is not installed; use NumpyANNIndex instead")
        self.item_ids = [str(item_id) for item_id in item_ids]
        vectors = embeddings.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
        vectors = vectors / norms
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        vector = query.astype(np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        scores, indices = self.index.search(vector.reshape(1, -1), top_k)
        return [
            (self.item_ids[int(index)], float(score))
            for score, index in zip(scores[0], indices[0], strict=False)
            if int(index) >= 0
        ]
