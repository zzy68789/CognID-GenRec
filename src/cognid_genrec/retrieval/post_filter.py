from __future__ import annotations

from collections.abc import Iterable


def filter_seen_items(candidates: Iterable[dict], seen_item_ids: Iterable[str]) -> list[dict]:
    seen = {str(item_id) for item_id in seen_item_ids}
    return [
        dict(candidate)
        for candidate in candidates
        if str(candidate["item_id"]) not in seen
    ]
