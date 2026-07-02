from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import pandas as pd


class DiversityReranker:
    def __init__(
        self,
        freshness_window_days: int = 7,
        freshness_boost: float = 0.15,
        max_per_author: int = 2,
        long_tail_slots: int = 1,
        long_tail_exposure_threshold: int = 5,
        long_tail_quality_threshold: float = 0.85,
        now: str | None = None,
    ) -> None:
        self.freshness_window_days = freshness_window_days
        self.freshness_boost = freshness_boost
        self.max_per_author = max_per_author
        self.long_tail_slots = long_tail_slots
        self.long_tail_exposure_threshold = long_tail_exposure_threshold
        self.long_tail_quality_threshold = long_tail_quality_threshold
        self.now = pd.Timestamp(now) if now else pd.Timestamp.utcnow().tz_localize(None)

    def rerank(
        self,
        candidates: Iterable[dict],
        items: pd.DataFrame,
        top_k: int,
    ) -> list[dict]:
        item_meta = {
            str(row["item_id"]): row
            for row in items.to_dict("records")
        }
        enriched = [
            self._enrich_candidate(candidate, item_meta)
            for candidate in candidates
            if str(candidate["item_id"]) in item_meta
        ]
        if not enriched:
            return []

        long_tail = sorted(
            [candidate for candidate in enriched if candidate["is_long_tail"]],
            key=lambda candidate: (
                -candidate["quality_score"],
                candidate["exposure_count"],
                candidate["item_id"],
            ),
        )
        reserved_long_tail = long_tail[: min(self.long_tail_slots, top_k)]
        reserved_ids = {candidate["item_id"] for candidate in reserved_long_tail}
        normal_candidates = [
            candidate for candidate in enriched if candidate["item_id"] not in reserved_ids
        ]
        normal_candidates.sort(
            key=lambda candidate: (
                -candidate["adjusted_score"],
                candidate["topic"],
                candidate["item_id"],
            )
        )

        selected: list[dict] = []
        author_counts: Counter[str] = Counter()
        normal_slots = top_k - len(reserved_long_tail)
        while len(selected) < normal_slots and normal_candidates:
            candidate = self._choose_next_candidate(
                normal_candidates,
                selected,
                author_counts,
            )
            if candidate is None:
                break
            selected.append(candidate)
            author_counts[candidate["author_id"]] += 1
            normal_candidates = [
                remaining
                for remaining in normal_candidates
                if remaining["item_id"] != candidate["item_id"]
            ]

        for candidate in reserved_long_tail:
            if len(selected) >= top_k:
                break
            if author_counts[candidate["author_id"]] >= self.max_per_author:
                continue
            protected = dict(candidate)
            protected["source_reason"] = "long_tail_protection"
            selected.append(protected)
            author_counts[protected["author_id"]] += 1

        return selected[:top_k]

    def _enrich_candidate(self, candidate: dict, item_meta: dict[str, dict]) -> dict:
        item_id = str(candidate["item_id"])
        meta = item_meta[item_id]
        publish_time = pd.Timestamp(meta["publish_time"])
        if publish_time.tzinfo is not None:
            publish_time = publish_time.tz_convert(None)
        is_fresh = publish_time >= self.now - pd.Timedelta(days=self.freshness_window_days)
        exposure_count = int(meta.get("exposure_count", 0) or 0)
        quality_score = float(meta.get("quality_score", 0.0) or 0.0)
        adjusted_score = float(candidate["score"]) + (self.freshness_boost if is_fresh else 0.0)
        return {
            **candidate,
            "item_id": item_id,
            "topic": str(meta["topic"]),
            "author_id": str(meta["author_id"]),
            "publish_time": str(meta["publish_time"]),
            "quality_score": quality_score,
            "exposure_count": exposure_count,
            "adjusted_score": adjusted_score,
            "is_fresh": is_fresh,
            "is_long_tail": (
                exposure_count <= self.long_tail_exposure_threshold
                and quality_score >= self.long_tail_quality_threshold
            ),
            "source_reason": "ranked",
        }

    def _choose_next_candidate(
        self,
        candidates: list[dict],
        selected: list[dict],
        author_counts: Counter[str],
    ) -> dict | None:
        allowed = [
            candidate
            for candidate in candidates
            if author_counts[candidate["author_id"]] < self.max_per_author
        ]
        if not allowed:
            return None
        if not selected:
            return allowed[0]
        previous_topic = selected[-1]["topic"]
        diversified = [
            candidate for candidate in allowed if candidate["topic"] != previous_topic
        ]
        return diversified[0] if diversified else allowed[0]
