from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from cognid_genrec.retrieval.semantic_id_decoder import SemanticIDDecoder


class GenerativeRetriever:
    def __init__(self, beam_width: int = 5, max_steps: int = 3) -> None:
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.item_to_semantic: dict[str, str] = {}
        self.transition_counts: dict[str, dict[str, float]] = {}
        self.global_counts: dict[str, float] = {}
        self.decoder = SemanticIDDecoder({})

    def fit(
        self,
        user_sequences: Iterable[dict],
        semantic_id_mapping: pd.DataFrame,
    ) -> "GenerativeRetriever":
        self.item_to_semantic = {
            str(row["item_id"]): str(row["semantic_id_str"])
            for row in semantic_id_mapping.to_dict("records")
        }
        self.decoder = SemanticIDDecoder.from_mapping(semantic_id_mapping)

        transitions: defaultdict[str, Counter[str]] = defaultdict(Counter)
        global_counts: Counter[str] = Counter()
        for sequence in user_sequences:
            semantic_sequence = [
                self.item_to_semantic[item_id]
                for item_id in map(str, sequence.get("item_ids", []))
                if item_id in self.item_to_semantic
            ]
            global_counts.update(semantic_sequence)
            for left, right in zip(semantic_sequence, semantic_sequence[1:], strict=False):
                transitions[left][right] += 1.0

        self.transition_counts = {
            source: dict(targets) for source, targets in transitions.items()
        }
        self.global_counts = {item: float(count) for item, count in global_counts.items()}
        return self

    def generate_semantic_ids(self, history_item_ids: Iterable[str]) -> list[str]:
        history_semantic_ids = [
            self.item_to_semantic[item_id]
            for item_id in map(str, history_item_ids)
            if item_id in self.item_to_semantic
        ]
        excluded = set(history_semantic_ids)
        if not history_semantic_ids:
            return [semantic_id for semantic_id, _ in self._global_candidates(excluded)]

        beams = [([], history_semantic_ids[-1], 0.0)]
        completed: list[tuple[list[str], float]] = []
        for _ in range(self.max_steps):
            next_beams: list[tuple[list[str], str, float]] = []
            for generated, current_semantic_id, score in beams:
                candidates = self._next_candidates(current_semantic_id, excluded | set(generated))
                for candidate, candidate_score in candidates:
                    next_beams.append(
                        (
                            [*generated, candidate],
                            candidate,
                            score + candidate_score,
                        )
                    )
            if not next_beams:
                completed.extend((generated, score) for generated, _, score in beams)
                break
            next_beams.sort(key=lambda beam: (-beam[2], beam[0]))
            beams = next_beams[: self.beam_width]
        else:
            completed.extend((generated, score) for generated, _, score in beams)

        if not completed:
            completed.extend((generated, score) for generated, _, score in beams)
        completed.sort(key=lambda beam: (-beam[1], beam[0]))
        best_path = completed[0][0] if completed else []
        if best_path:
            return best_path
        return [semantic_id for semantic_id, _ in self._global_candidates(excluded)]

    def recommend(self, history_item_ids: Iterable[str], top_k: int = 10) -> list[str]:
        history = [str(item_id) for item_id in history_item_ids]
        generated_semantic_ids = self.generate_semantic_ids(history)
        recommendations = self.decoder.decode(
            generated_semantic_ids,
            top_k=top_k,
            exclude_item_ids=history,
        )
        if len(recommendations) >= top_k:
            return recommendations

        fallback_semantic_ids = [
            semantic_id
            for semantic_id, _ in self._global_candidates(set(generated_semantic_ids))
        ]
        fallback = self.decoder.decode(
            fallback_semantic_ids,
            top_k=top_k,
            exclude_item_ids=set(history) | set(recommendations),
        )
        return [*recommendations, *fallback][:top_k]

    def save(self, path: str | Path) -> None:
        payload = {
            "beam_width": self.beam_width,
            "max_steps": self.max_steps,
            "item_to_semantic": self.item_to_semantic,
            "transition_counts": self.transition_counts,
            "global_counts": self.global_counts,
            "semantic_to_items": self.decoder.to_dict(),
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GenerativeRetriever":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        retriever = cls(
            beam_width=int(payload["beam_width"]),
            max_steps=int(payload["max_steps"]),
        )
        retriever.item_to_semantic = {
            str(item_id): str(semantic_id)
            for item_id, semantic_id in payload["item_to_semantic"].items()
        }
        retriever.transition_counts = {
            str(source): {str(target): float(score) for target, score in targets.items()}
            for source, targets in payload["transition_counts"].items()
        }
        retriever.global_counts = {
            str(semantic_id): float(count)
            for semantic_id, count in payload["global_counts"].items()
        }
        retriever.decoder = SemanticIDDecoder.from_dict(payload["semantic_to_items"])
        return retriever

    def _next_candidates(
        self,
        semantic_id: str,
        excluded: set[str],
    ) -> list[tuple[str, float]]:
        candidates = [
            (candidate, score)
            for candidate, score in self.transition_counts.get(semantic_id, {}).items()
            if candidate not in excluded
        ]
        if not candidates:
            return self._global_candidates(excluded)
        return sorted(candidates, key=lambda item_score: (-item_score[1], item_score[0]))

    def _global_candidates(self, excluded: set[str]) -> list[tuple[str, float]]:
        return sorted(
            (
                (semantic_id, score)
                for semantic_id, score in self.global_counts.items()
                if semantic_id not in excluded
            ),
            key=lambda item_score: (-item_score[1], item_score[0]),
        )
