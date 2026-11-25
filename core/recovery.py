"""Identity recovery: deciding whether a new detection is an *old* identity.

This is the heart of the problem statement.  A naive "closest appearance match"
produces false identities, which is worse than losing a track.  So a recovery
proposal has to clear five independent gates:

1. class gate       -- same semantic class.
2. time gate        -- the identity has actually been gone for a moment.
3. motion gate      -- inside a search region predicted from the last known
                       position and velocity, growing with time-absent.
4. appearance gate  -- *multi-cue* similarity (each appearance block scored
                       separately, then fused) above a threshold that rises
                       with time-absent and with a weak appearance gallery.
                       A single cosine over the whole vector let a shared
                       jacket colour outvote the pattern cue: two different
                       actors scored 0.90.  Fused, the same impostor scores
                       0.79 while the true actor still scores 0.90.
5. ambiguity gate   -- the leader must beat *everything else that could explain
                       the same pixels*: every other known identity (active,
                       lost or archived) and every other detection competing
                       for this identity, by a margin.

Gate 5 used to compare only against other *lost* identities.  That is exactly
the case the benchmark exposed: while a look-alike was being tracked live, a
recovery for its twin scored 0.787 while the live twin scored 0.780 -- a 0.007
lead.  Nothing consulted the live twin, so identity AX-001 was handed to a
different actor for 57 frames.  Rivals are now drawn from the whole identity
memory, which is what makes a swap fail even when both assignments look
self-consistent on their own.

Assignments are solved jointly with the Hungarian algorithm, so two detections
can never both claim the same identity in a single frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from .appearance import CUE_SIZES, CUE_WEIGHTS, cosine
from .config import ARGUSConfig


@dataclass
class RecoveryProposal:
    global_id: str
    det_index: int
    score: float
    margin: float
    distance_ratio: float
    threshold: float
    frames_absent: int
    rival_id: str = ""
    rival_score: float = 0.0

    def describe(self) -> str:
        rival = f" rival={self.rival_id} {self.rival_score:.3f}" if self.rival_id else ""
        return (
            f"app={self.score:.3f} thr={self.threshold:.2f} "
            f"margin={self.margin:.3f} gap={self.frames_absent}f "
            f"dist={self.distance_ratio:.2f}R{rival}"
        )


class IdentityRecovery:
    def __init__(self, memory, reid, config: ARGUSConfig = None):
        self.memory = memory
        self.reid = reid
        self.cfg = config or ARGUSConfig()
        self._last_reject = {}

    # -- gates --------------------------------------------------------------------
    def _threshold(self, identity, frames_absent: int) -> float:
        drift = 0.055 * min(1.0, frames_absent / 120.0)
        weak_gallery = 0.05 if len(identity.gallery) < 3 else 0.0
        return float(min(self.cfg.recovery_threshold + drift + weak_gallery, 0.96))

    def _search_radius(self, identity, frames_absent: int) -> float:
        width, height = identity.size
        body = max(height, width)
        growth = 0.9 + 0.60 * min(1.0, frames_absent / 120.0)
        return max(56.0, self.cfg.search_radius_scale * body * growth)

    def _predicted_center(self, identity, frames_absent: int):
        cx, cy = identity.center
        if identity.velocity is not None:
            weight = 1.0 / (1.0 + frames_absent / 45.0)   # velocity goes stale
            cx += float(identity.velocity[0]) * frames_absent * weight
            cy += float(identity.velocity[1]) * frames_absent * weight
        return cx, cy

    def _motion_gate(self, identity, bbox, frames_absent: int):
        cx, cy = self._predicted_center(identity, frames_absent)
        bx = (float(bbox[0]) + float(bbox[2])) / 2.0
        by = (float(bbox[1]) + float(bbox[3])) / 2.0
        distance = float(np.hypot(bx - cx, by - cy))
        radius = self._search_radius(identity, frames_absent)
        height = max(float(bbox[3]) - float(bbox[1]), 1.0)
        scale_ratio = height / max(identity.size[1], 1.0)
        scale_ok = 0.45 <= scale_ratio <= 2.3
        return distance <= radius, distance / radius, scale_ok

    # -- similarity ---------------------------------------------------------------
    def _similarity_matrix(self, detections: list, known: list) -> np.ndarray:
        """Fused per-cue similarity of every crop against every identity gallery.

        Vectorised on purpose: the rival scan compares a crop with the entire
        identity memory, so a per-pair Python loop would dominate the frame
        budget once dozens of identities have been created.  Each cue block is
        normalised and scored on its own before the weighted sum, so a shared
        colour cannot swamp the pattern and texture evidence.
        """
        n_det, n_known = len(detections), len(known)
        similarity = np.full((n_det, n_known), -1.0, dtype=float)

        dims = [int(np.asarray(det.embedding).reshape(-1).size)
                for det in detections if det.embedding is not None]
        if not dims:
            return similarity
        dim = dims[0]
        rows = []
        for det in detections:
            vector = (np.asarray(det.embedding, dtype=np.float32).reshape(-1)
                      if det.embedding is not None else np.zeros(0, dtype=np.float32))
            rows.append(vector if vector.size == dim else np.zeros(dim, dtype=np.float32))
        det_rows = np.stack(rows)

        columns, owners = [], []
        for k, identity in enumerate(known):
            views = [np.asarray(view, dtype=np.float32) for view in identity.gallery]
            if identity.embedding is not None:
                views.append(np.asarray(identity.embedding, dtype=np.float32))
            views = [view for view in views if view.shape == det_rows.shape[1:]]
            if not views:
                continue
            columns.append(np.stack(views))
            owners.extend([k] * len(views))
        if not columns:
            return similarity

        gallery = np.concatenate(columns, axis=0)
        raw = np.zeros((n_det, gallery.shape[0]), dtype=float)
        offset = 0
        for size, weight in zip(CUE_SIZES, CUE_WEIGHTS):
            view_block = gallery[:, offset:offset + size]
            probe_block = det_rows[:, offset:offset + size]
            view_block = view_block / (np.linalg.norm(view_block, axis=1, keepdims=True) + 1e-9)
            probe_block = probe_block / (np.linalg.norm(probe_block, axis=1, keepdims=True) + 1e-9)
            raw += weight * (probe_block @ view_block.T)
            offset += size
        raw = np.clip(raw, -1.0, 1.0)
        owners = np.asarray(owners, dtype=int)

        for i, det in enumerate(detections):
            if det.embedding is None or np.linalg.norm(det_rows[i]) < 1e-9:
                continue
            scores = raw[i]
            for k, identity in enumerate(known):
                if identity.class_name != det.class_name:
                    continue
                own = scores[owners == k]
                if own.size:
                    similarity[i, k] = float(np.max(own))
        return similarity

    # -- matching -----------------------------------------------------------------
    def plausible(self, detection, frame_idx: int, exclude_ids=(), bar: float = None) -> bool:
        """Is any waiting identity close enough to be worth waiting for?

        Used before a brand-new track mints an identity: when a lost identity is
        plausibly this object, the track stays unidentified for a few frames so
        a cleaner crop can claim it instead of duplicating it.
        """
        excluded = set(exclude_ids)
        candidates = [
            identity
            for identity in self.memory.get_lost(frame_idx, self.cfg.recovery_min_gap)
            if identity.global_id not in excluded and identity.embedding is not None
        ]
        if not candidates or detection.embedding is None:
            return False
        known = self.memory.known()
        if not known:
            return False
        threshold = self.cfg.recovery_threshold - 0.12 if bar is None else bar
        similarity = self._similarity_matrix([detection], known)
        column = {identity.global_id: k for k, identity in enumerate(known)}
        return any(
            similarity[0, column[c.global_id]] >= threshold
            for c in candidates if c.global_id in column
        )

    def propose(self, detections: list, frame_idx: int, exclude_ids=()) -> List[RecoveryProposal]:
        if not detections:
            return []

        excluded = set(exclude_ids)
        candidates = [
            identity
            for identity in self.memory.get_lost(frame_idx, self.cfg.recovery_min_gap)
            if identity.global_id not in excluded and identity.embedding is not None
        ]
        if not candidates:
            return []

        # Every known identity is a possible rival -- including the ones that are
        # being tracked right now.
        known = self.memory.known()
        if not known:
            return []
        column = {identity.global_id: k for k, identity in enumerate(known)}
        candidates = [identity for identity in candidates if identity.global_id in column]
        if not candidates:
            return []

        similarity = self._similarity_matrix(detections, known)

        n_det, n_cand = len(detections), len(candidates)
        scores = np.zeros((n_det, n_cand), dtype=float)
        motion_ratio = np.ones((n_det, n_cand), dtype=float)
        valid = np.zeros((n_det, n_cand), dtype=bool)
        thresholds = np.zeros(n_cand, dtype=float)

        for j, identity in enumerate(candidates):
            frames_absent = frame_idx - identity.last_seen_frame
            thresholds[j] = self._threshold(identity, frames_absent)
            for i, det in enumerate(detections):
                score = similarity[i, column[identity.global_id]]
                if det.class_name != identity.class_name or det.embedding is None:
                    continue
                if score < thresholds[j]:
                    continue
                motion_ok, ratio, scale_ok = self._motion_gate(identity, det.bbox, frames_absent)
                if not motion_ok or not scale_ok:
                    continue
                scores[i, j] = score
                motion_ratio[i, j] = ratio
                valid[i, j] = True

        if not valid.any():
            return []

        cost = np.where(valid, (1.0 - scores) + 0.12 * motion_ratio, 1e6)
        rows, cols = linear_sum_assignment(cost)

        proposals: List[RecoveryProposal] = []
        for i, j in zip(rows, cols):
            if not valid[i, j]:
                continue

            identity = candidates[j]
            own_column = column[identity.global_id]

            # Rival A: any other identity -- active or not -- that could explain
            # this same detection.  This is what catches look-alike swaps.
            rival_score, rival_id = 0.0, ""
            for k, other in enumerate(known):
                if k == own_column:
                    continue
                if similarity[i, k] > rival_score:
                    rival_score, rival_id = float(similarity[i, k]), other.global_id

            # Rival B: another detection competing for this identity.
            other_dets = max(
                (scores[k, j] for k in range(n_det) if k != i and valid[k, j]), default=0.0
            )

            best_rival = max(rival_score, float(other_dets))
            margin = float(scores[i, j] - best_rival)

            if margin < self.cfg.reid_margin:
                # One entry per identity per episode, not one per frame.
                last = self._last_reject.get(identity.global_id, -999)
                if frame_idx - last >= 20:
                    self._last_reject[identity.global_id] = frame_idx
                    self.memory._log(frame_idx, "RECOVERY_REJECTED", identity.global_id,
                                     float(scores[i, j]),
                                     f"ambiguous: {scores[i, j]:.3f} vs "
                                     f"{(rival_id or 'competing detection')} {best_rival:.3f}")
                continue

            proposals.append(
                RecoveryProposal(
                    global_id=identity.global_id,
                    det_index=int(i),
                    score=float(scores[i, j]),
                    margin=margin,
                    distance_ratio=float(motion_ratio[i, j]),
                    threshold=float(thresholds[j]),
                    frames_absent=int(frame_idx - identity.last_seen_frame),
                    rival_id=rival_id,
                    rival_score=rival_score,
                )
            )
        return proposals
