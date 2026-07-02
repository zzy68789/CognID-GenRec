from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import pandas as pd


class SemanticIDDecoder:
    def __init__(self, semantic_to_items: dict[str, list[str]]) -> None:
        self.semantic_to_items = {
            semantic_id: sorted(item_ids)
            for semantic_id, item_ids in semantic_to_items.items()
        }

    @classmethod
    def from_mapping(cls, mapping: pd.DataFrame) -> "SemanticIDDecoder":
        semantic_to_items: defaultdict[str, list[str]] = defaultdict(list)
        for row in mapping.to_dict("records"):
            semantic_id = normalize_semantic_id(row.get("semantic_id_str", row.get("semantic_id")))
            semantic_to_items[semantic_id].append(str(row["item_id"]))
        return cls(dict(semantic_to_items))

    @classmethod
    def from_dict(cls, payload: dict[str, list[str]]) -> "SemanticIDDecoder":
        return cls(payload)

    def to_dict(self) -> dict[str, list[str]]:
        return self.semantic_to_items

    def decode(
        self,
        semantic_ids: Iterable[str | list[int] | tuple[int, ...]],
        top_k: int = 10,
        exclude_item_ids: Iterable[str] | None = None,
    ) -> list[str]:
        excluded = set(exclude_item_ids or [])
        decoded: list[str] = []
        seen: set[str] = set()
        for semantic_id in semantic_ids:
            semantic_id_str = normalize_semantic_id(semantic_id)
            for item_id in self.semantic_to_items.get(semantic_id_str, []):
                if item_id in excluded or item_id in seen:
                    continue
                decoded.append(item_id)
                seen.add(item_id)
                if len(decoded) >= top_k:
                    return decoded
        return decoded


def normalize_semantic_id(semantic_id: str | list[int] | tuple[int, ...] | object) -> str:
    if isinstance(semantic_id, str):
        return semantic_id
    if isinstance(semantic_id, (list, tuple)):
        return "-".join(map(str, semantic_id))
    return str(semantic_id)
