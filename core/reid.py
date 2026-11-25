"""Appearance matching (re-ID) utilities.

The heavy lifting is done by :mod:`core.appearance`.  What matters here is
*how* a match is judged: a raw cosine score is never enough, because two people
in similar clothes both score high.  Every decision therefore also inspects the
margin over the runner-up identity.
"""

from __future__ import annotations

import numpy as np

from .appearance import cosine


class ReID:
    def __init__(self, default_threshold: float = 0.72):
        self.default_threshold = default_threshold

    def cosine_similarity(self, a, b) -> float:
        return cosine(a, b)

    def compare(self, old_features, new_features) -> dict:
        score = cosine(old_features, new_features)
        return {"similarity": round(score, 4), "match": score >= self.default_threshold}

    def rank(self, embedding, candidates: dict) -> list:
        scored = [(key, cosine(embedding, value)) for key, value in candidates.items()]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def best_match(self, embedding, candidates: dict, threshold: float = None,
                   margin: float = 0.0):
        """Return (key, score, accepted).  ``accepted`` enforces the margin gate."""
        if embedding is None or not candidates:
            return None, 0.0, False
        threshold = self.default_threshold if threshold is None else threshold
        ranked = self.rank(embedding, candidates)
        key, score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        return key, float(score), bool(score >= threshold and (score - runner_up) >= margin)


def normalize_embedding(embedding):
    if embedding is None:
        return None
    array = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm < 1e-9:
        return None
    return array / norm
