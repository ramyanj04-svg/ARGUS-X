"""The ARGUS-X tracking engine.

One call to :meth:`TrackManager.step` performs a full perception cycle:

    camera compensation -> Kalman prediction -> appearance embedding ->
    two-stage association (high / low confidence) -> track lifecycle ->
    identity recovery -> new track spawning -> snapshot for rendering

Design notes that matter for the problem statement
--------------------------------------------------
* An identity is created only after ``n_init`` consecutive hits, so detector
  noise cannot invent identities.  During that time the track is drawn as a
  *provisional* track (``T-xx``) and is not written to memory.
* A lost track is kept alive by the Kalman filter and by a growing appearance
  requirement -- the longer the object has been missing, the stronger the
  evidence must be, which is how we avoid false re-identifications.
* Once a track is gone beyond ``max_age``, its identity stays in memory and
  can only come back through :class:`core.recovery.IdentityRecovery`, which
  requires a spatial gate, a strict appearance gate and a margin over every
  competing identity.  Recovery is provisional for ``recovery_confirm`` frames
  and is rolled back if it does not hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .appearance import AppearanceEncoder, cosine
from .association import INFEASIBLE, greedy_solve, solve
from .config import ARGUSConfig
from .detector import Detection
from .memory import IdentityMemory
from .motion import CameraMotionEstimator, KalmanBoxTracker, iou_batch
from .recovery import IdentityRecovery
from .reid import ReID


# --------------------------------------------------------------------------- track
class Track:
    """A short-lived motion hypothesis.  An identity may outlive many tracks."""

    _counter = 0

    def __init__(self, detection: Detection, config: ARGUSConfig, frame_idx: int,
                 global_id: Optional[str] = None, recovered: bool = False,
                 recovery_score: float = 0.0):
        Track._counter += 1
        self.track_no = Track._counter
        self.cfg = config
        self.kf = KalmanBoxTracker(detection.bbox)
        self.bbox = np.asarray(detection.bbox, dtype=float)
        self.class_name = detection.class_name
        self.det_score = float(detection.score)
        self.embedding = detection.embedding
        self.bank: List[np.ndarray] = []
        if detection.embedding is not None:
            self.bank.append(np.asarray(detection.embedding, dtype=np.float32))
        self.quality = float(detection.quality)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.created_frame = frame_idx
        self.global_id = global_id
        self.recovered = recovered
        self.recovery_score = float(recovery_score)
        self.state = "RECOVERING" if recovered else "TENTATIVE"
        self.verified = not recovered
        self.miss_count = 0
        self.confidence = 0.0
        self.deferred = False       # waiting for a cleaner view before minting an ID

    # -- geometry --------------------------------------------------------------
    @property
    def display_id(self) -> str:
        return self.global_id if self.global_id else f"T-{self.track_no:02d}"

    @property
    def is_confirmed(self) -> bool:
        return self.state == "CONFIRMED"

    # -- updates ---------------------------------------------------------------
    def predict(self) -> None:
        self.age += 1
        self.kf.predict()
        self.bbox = self.kf.bbox

    def shift(self, dx: float, dy: float) -> None:
        self.kf.shift(dx, dy)
        self.bbox = self.kf.bbox

    def update(self, detection: Detection, frame_idx: int) -> None:
        self.kf.update(detection.bbox)
        self.bbox = self.kf.bbox
        self.det_score = 0.7 * self.det_score + 0.3 * float(detection.score)
        self.quality = 0.8 * self.quality + 0.2 * float(detection.quality)
        self.hits += 1
        self.time_since_update = 0
        self.miss_count = 0

        if detection.embedding is not None:
            if self.embedding is None:
                self.embedding = detection.embedding
            else:
                merged = 0.88 * np.asarray(self.embedding, dtype=np.float32) + 0.12 * np.asarray(
                    detection.embedding, dtype=np.float32
                )
                norm = float(np.linalg.norm(merged))
                self.embedding = (merged / norm).astype(np.float32) if norm > 1e-9 else self.embedding
            if detection.quality >= 0.25:
                self.bank.append(np.asarray(detection.embedding, dtype=np.float32))
                if len(self.bank) > 6:
                    del self.bank[0]

    def similarity(self, embedding) -> float:
        if embedding is None:
            return 0.0
        scores = [cosine(embedding, crop) for crop in self.bank] if self.bank else []
        if self.embedding is not None:
            scores.append(cosine(embedding, self.embedding))
        return float(max(scores)) if scores else 0.0

    def confidence_now(self, appearance_similarity: float = 0.0) -> float:
        recency = 1.0 / (1.0 + self.time_since_update)
        base = 0.55 * self.det_score + 0.25 * appearance_similarity + 0.20 * recency
        if self.state == "TENTATIVE":
            base *= min(1.0, 0.45 + 0.18 * self.hits)
        return float(np.clip(base, 0.0, 1.0))


@dataclass
class FrameResult:
    frame_index: int
    boxes: List[dict] = field(default_factory=list)
    ghosts: List[dict] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)
    detections: int = 0
    active: int = 0
    provisional: int = 0
    camera_shift: tuple = (0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "frame": self.frame_index,
            "boxes": self.boxes,
            "ghosts": self.ghosts,
            "detections": self.detections,
            "active": self.active,
            "provisional": self.provisional,
        }


# ------------------------------------------------------------------------- manager
class TrackManager:
    def __init__(self, config: ARGUSConfig = None, detector=None):
        self.cfg = config or ARGUSConfig()
        self.detector = detector
        self.encoder = AppearanceEncoder(min_crop_px=self.cfg.min_crop_px)
        self.reid = ReID(self.cfg.reid_threshold)
        self.memory = IdentityMemory(
            gallery_size=self.cfg.gallery_size,
            archive_after=self.cfg.archive_after,
        )
        self.recovery = IdentityRecovery(self.memory, self.reid, self.cfg)
        self.camera = CameraMotionEstimator() if self.cfg.camera_compensation else None
        self.tracks: List[Track] = []
        self.frame_index = 0
        self.stats = {
            "detections": 0,
            "recoveries": 0,
            "rejected_recoveries": 0,
            "duplicate_suppressed": 0,
            "new_identities": 0,
            "camera_frames": 0,
        }
        self._event_cursor = 0

    # -- main loop ---------------------------------------------------------------
    def step(self, frame, detections: List[Detection]) -> FrameResult:
        self.frame_index += 1
        idx = self.frame_index
        cfg = self.cfg

        # 1) global camera motion ------------------------------------------------
        dx = dy = 0.0
        if self.camera is not None:
            dx, dy = self.camera.estimate(frame)
            if abs(dx) > 1.0 or abs(dy) > 1.0:
                self.stats["camera_frames"] += 1
                for track in self.tracks:
                    track.shift(dx, dy)

        # 2) motion prediction ---------------------------------------------------
        for track in self.tracks:
            track.predict()
            if track.time_since_update > 6:
                track.kf.damp_velocity(cfg.velocity_decay)

        # 3) appearance embeddings ----------------------------------------------
        detections = list(detections)
        for det in detections:
            if det.embedding is None:
                det.embedding = self.encoder.encode(frame, det.bbox)
            if det.quality <= 0.0:
                det.quality = self.encoder.quality(frame, det.bbox, det.score)
        self.stats["detections"] += len(detections)

        # 4) association ---------------------------------------------------------
        matched_tracks, matched_dets = set(), set()
        high = [i for i, d in enumerate(detections) if d.score >= cfg.high_thresh]
        low = [i for i, d in enumerate(detections) if cfg.conf_thresh <= d.score < cfg.high_thresh]

        pairs = self._associate(self.tracks, high, detections, strict=False)
        for ti, di in pairs:
            matched_tracks.add(ti)
            matched_dets.add(high[di])

        remaining_tracks = [t for i, t in enumerate(self.tracks) if i not in matched_tracks]
        low_pairs = self._associate(remaining_tracks, low, detections, strict=True)
        for ti, di in low_pairs:
            track = remaining_tracks[ti]
            matched_tracks.add(self.tracks.index(track))
            matched_dets.add(low[di])

        # 5) apply matches -------------------------------------------------------
        for ti, di in pairs:
            self._on_match(self.tracks[ti], detections[high[di]], idx)
        for ti, di in low_pairs:
            self._on_match(remaining_tracks[ti], detections[low[di]], idx)

        # 6) unmatched tracks: occlusion bookkeeping ------------------------------
        survivors: List[Track] = []
        for ti, track in enumerate(self.tracks):
            if ti in matched_tracks:
                survivors.append(track)
                continue
            track.time_since_update += 1
            track.miss_count += 1
            track.bbox = track.kf.bbox

            if track.state == "RECOVERING":
                # Recovery verification tolerates a single dropped detection,
                # but two consecutive misses mean the match did not hold.
                if track.time_since_update > 1:
                    if track.global_id:
                        self.memory.rollback(track.global_id, idx, track.recovery_score)
                        self.stats["rejected_recoveries"] += 1
                    continue
                survivors.append(track)
                continue

            if track.global_id and track.time_since_update == 1:
                self.memory.mark_lost(track.global_id, idx, "no detection")

            # Short gap: keep the motion hypothesis alive.  Long gap: kill the
            # track and leave the identity in memory, so that re-acquisition
            # must pass the strict, verified recovery gates.  This is what
            # stops a drifting ghost box from stealing a different person.
            if track.time_since_update > min(cfg.max_age, cfg.reid_after):
                continue

            survivors.append(track)
        self.tracks = survivors

        # 7) identity recovery for leftover detections ---------------------------
        leftover = [i for i in range(len(detections)) if i not in matched_dets]
        live_ids = {t.global_id for t in self.tracks if t.global_id}
        live_tracks = [t for t in self.tracks if t.global_id and t.time_since_update == 0]
        if leftover:
            proposals = self.recovery.propose(
                [detections[i] for i in leftover], idx, exclude_ids=live_ids
            )
            for proposal in proposals:
                det = detections[leftover[proposal.det_index]]

                # Active-claim guard: if a currently tracked identity explains
                # this detection at least as well, the object already has an ID
                # and this recovery would duplicate it (false match).
                claimed = False
                for live in live_tracks:
                    if live.similarity(det.embedding) >= max(
                        self.cfg.app_gate, proposal.score - self.cfg.reid_margin
                    ):
                        claimed = True
                        break
                    if float(iou_batch(np.array([live.bbox]), np.array([det.bbox]))[0, 0]) >= 0.5:
                        claimed = True
                        break
                if not claimed:
                    occupied = [t.bbox for t in self.tracks if t.global_id]
                    if occupied:
                        overlap = float(np.max(iou_batch(
                            np.array(occupied, dtype=float), np.array([det.bbox], dtype=float)
                        )))
                        claimed = overlap >= 0.30
                if claimed:
                    # Not a failed recovery: the object simply already has an
                    # identity (or a track is sitting on it), so this proposal
                    # would duplicate it.  Counted separately, never logged
                    # per frame.
                    self.stats["duplicate_suppressed"] += 1
                    continue

                track = self._spawn(det, idx, global_id=proposal.global_id,
                                    recovered=True, recovery_score=proposal.score)
                if track is not None:
                    self.memory._log(idx, "RECOVERY_CANDIDATE", proposal.global_id,
                                     proposal.score, proposal.describe())
                    matched_dets.add(leftover[proposal.det_index])

        # 8) brand new tracks ----------------------------------------------------
        leftover = [i for i in range(len(detections)) if i not in matched_dets]
        for i in leftover:
            det = detections[i]
            if det.score < cfg.new_track_thresh or det.embedding is None:
                continue
            if det.quality < 0.12:
                continue
            if self._is_duplicate(det):
                continue
            self._spawn(det, idx)

        # 9) housekeeping --------------------------------------------------------
        self.memory.expire(idx)
        result = self._snapshot(idx, detections, (dx, dy))

        new_events = self.memory.events[self._event_cursor:]
        self._event_cursor = len(self.memory.events)
        result.events = new_events
        self.stats["new_identities"] = len(self.memory.identities)
        self.stats["recoveries"] = sum(i.recovery_count for i in self.memory.identities.values())
        return result

    # -- association -------------------------------------------------------------
    def _associate(self, track_pool: List[Track], det_indices: List[int],
                   detections: List[Detection], strict: bool = False):
        if not track_pool or not det_indices:
            return []

        track_boxes = np.array([t.bbox for t in track_pool], dtype=float)
        det_boxes = np.array([detections[i].bbox for i in det_indices], dtype=float)
        iou = iou_batch(track_boxes, det_boxes)

        similarity = np.zeros_like(iou)
        for a, track in enumerate(track_pool):
            for b, det_index in enumerate(det_indices):
                similarity[a, b] = track.similarity(detections[det_index].embedding)

        cost = self.cfg.w_iou * (1.0 - iou) + self.cfg.w_app * (1.0 - similarity)
        valid = np.ones_like(iou, dtype=bool)

        centres = np.array([[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, b[3] - b[1]]
                            for b in det_boxes], dtype=float)

        for a, track in enumerate(track_pool):
            gap = track.time_since_update
            # The longer the gap, the stronger the appearance evidence needed.
            if gap == 0:
                app_need = self.cfg.app_gate
            else:
                app_need = min(0.84, self.cfg.reappear_app_gate + 0.0018 * gap)
            iou_need = self.cfg.iou_gate if gap <= 2 else 0.02
            if strict:
                app_need = max(app_need, self.cfg.low_conf_app_gate)

            track_box = track.bbox
            track_height = max(float(track_box[3] - track_box[1]), 1.0)
            track_centre = ((float(track_box[0]) + float(track_box[2])) / 2.0,
                            (float(track_box[1]) + float(track_box[3])) / 2.0)

            for b, det_index in enumerate(det_indices):
                det = detections[det_index]
                if det.class_name != track.class_name:
                    valid[a, b] = False
                    continue
                motion_ok = iou[a, b] >= iou_need
                appearance_ok = similarity[a, b] >= app_need
                if gap > 0:
                    # A stale prediction can overlap anyone; require plausible
                    # distance and scale before appearance is even considered.
                    distance = float(np.hypot(centres[b, 0] - track_centre[0],
                                              centres[b, 1] - track_centre[1]))
                    scale = centres[b, 2] / track_height
                    if distance > 2.2 * track_height or not (0.5 <= scale <= 2.0):
                        valid[a, b] = False
                        continue
                if strict:
                    motion_ok = iou[a, b] >= 0.30
                if gap == 0:
                    # Same frame: either strong motion overlap, or appearance
                    # support with at least a little overlap.  A weak IoU alone
                    # must never update a confirmed identity.
                    accepted = (iou[a, b] >= 0.30) or (appearance_ok and iou[a, b] >= 0.05)
                else:
                    # After a gap, motion is only a prediction, so appearance
                    # must confirm it.  Without this, random false positives
                    # get absorbed into lost tracks.
                    accepted = appearance_ok and motion_ok
                if strict:
                    # Low-confidence detections are only trusted with both
                    # strong motion and strong appearance evidence.
                    accepted = accepted and iou[a, b] >= 0.40 and similarity[a, b] >= app_need
                if not accepted:
                    valid[a, b] = False
                if gap > 0:
                    # Prefer motion-consistent pairs when evidence is thin.
                    cost[a, b] = 0.35 * (1.0 - iou[a, b]) + 0.65 * (1.0 - similarity[a, b])

        gated = np.where(valid, cost, INFEASIBLE)
        pairs = solve(cost, valid)
        if not pairs and valid.any():
            pairs = greedy_solve(cost, valid)
        return pairs

    # -- track lifecycle ---------------------------------------------------------
    def _spawn(self, detection: Detection, frame_idx: int, global_id: Optional[str] = None,
               recovered: bool = False, recovery_score: float = 0.0) -> Optional[Track]:
        track = Track(detection, self.cfg, frame_idx, global_id=global_id,
                      recovered=recovered, recovery_score=recovery_score)
        if self._is_duplicate(detection):
            return None
        self.tracks.append(track)
        return track

    def _claim_lost_identity(self, track: Track, detection: Detection,
                            frame_idx: int) -> bool:
        """Let an unidentified track adopt a waiting identity, if evidence allows."""
        live_ids = {t.global_id for t in self.tracks if t.global_id}
        proposals = self.recovery.propose([detection], frame_idx, exclude_ids=live_ids)
        if not proposals:
            return False
        proposal = proposals[0]

        # Same duplicate guard as the recovery path: never hand one identity to
        # a box that another identified track is already sitting on.
        occupied = [t.bbox for t in self.tracks if t.global_id and t is not track]
        if occupied:
            overlap = float(np.max(iou_batch(
                np.array(occupied, dtype=float), np.array([detection.bbox], dtype=float)
            )))
            if overlap >= 0.30:
                return False

        track.global_id = proposal.global_id
        track.state = "RECOVERING"
        track.recovered = True
        track.recovery_score = proposal.score
        track.hits = 1              # verification counts from this frame
        track.verified = False
        self.memory._log(frame_idx, "RECOVERY_CANDIDATE", proposal.global_id,
                         proposal.score, proposal.describe())
        return True

    def _is_duplicate(self, detection: Detection) -> bool:
        boxes = [t.bbox for t in self.tracks]
        if not boxes:
            return False
        ious = iou_batch(np.array(boxes, dtype=float), np.array([detection.bbox], dtype=float)).ravel()
        return bool(np.max(ious) >= self.cfg.duplicate_iou)

    def _on_match(self, track: Track, detection: Detection, frame_idx: int) -> None:
        appearance = track.similarity(detection.embedding)
        track.update(detection, frame_idx)
        track.confidence = track.confidence_now(appearance)

        if track.state == "RECOVERING":
            if track.hits >= self.cfg.recovery_confirm:
                track.state = "CONFIRMED"
                track.verified = True
                identity = self.memory.get(track.global_id)
                if identity is not None:
                    self.memory.observe(track.global_id, track.bbox, track.embedding,
                                        frame_idx, track.quality)
                    self.memory.mark_recovered(
                        track.global_id, frame_idx, track.recovery_score,
                        f"verified over {track.hits} frames",
                    )
            return

        if track.state == "TENTATIVE":
            if track.hits >= self.cfg.n_init:
                # Before minting a new identity, give this track the chance to
                # claim one that is waiting in memory.  An actor emerging from
                # behind an occluder is often clipped for a few frames; a new
                # identity created on that crop can never be undone, and it
                # duplicates an identity that was about to become matchable.
                # The wait only happens when something is actually waiting to
                # be claimed, so a genuinely new object is not delayed.
                live_ids = {t.global_id for t in self.tracks if t.global_id}
                if not track.deferred:
                    track.deferred = self.recovery.plausible(
                        detection, frame_idx, exclude_ids=live_ids)
                if track.deferred:
                    if self._claim_lost_identity(track, detection, frame_idx):
                        return
                    if track.age <= self.cfg.identity_grace:
                        # One clipped frame must not decide: the deferral is
                        # sticky until the grace expires.
                        return  # stay unidentified and try again next frame
                identity = self.memory.create(
                    track.bbox, track.embedding, track.class_name, frame_idx,
                    quality=min(1.0, track.quality),
                )
                track.global_id = identity.global_id
                track.state = "CONFIRMED"
            return

        # Confirmed track: write the observation into identity memory.
        if track.global_id:
            identity = self.memory.get(track.global_id)
            was_lost = identity is not None and identity.state != "ACTIVE"
            gap = (frame_idx - identity.last_seen_frame) if identity is not None else 0
            self.memory.observe(track.global_id, track.bbox, track.embedding,
                                frame_idx, track.quality,
                                update_appearance=True)
            # A short gap (< reid_after) is bridged by the motion model and is
            # not advertised as a re-identification.  Only the verified, long
            # horizon recovery path increments ``recovery_count``.
            if was_lost and gap > self.cfg.reid_after:
                self.memory.mark_recovered(track.global_id, frame_idx,
                                           track.confidence,
                                           f"motion model bridge over {gap} frames")

    # -- rendering snapshot ------------------------------------------------------
    def _snapshot(self, idx: int, detections: List[Detection], shift) -> FrameResult:
        result = FrameResult(frame_index=idx, detections=len(detections), camera_shift=shift)
        for track in self.tracks:
            if track.state == "TENTATIVE":
                result.provisional += 1
                result.boxes.append({
                    "id": track.display_id,
                    "bbox": [round(float(v), 1) for v in track.bbox],
                    "state": "PROVISIONAL",
                    "confidence": round(float(track.confidence), 3),
                    "recovered": False,
                })
                continue

            if track.time_since_update > 0:
                if self.cfg.draw_ghosts:
                    result.ghosts.append({
                        "id": track.display_id,
                        "bbox": [round(float(v), 1) for v in track.bbox],
                        "state": "LOST",
                        "confidence": round(float(track.confidence), 3),
                        "missing": int(track.time_since_update),
                        "recovered": track.recovered,
                    })
                continue

            result.active += 1
            result.boxes.append({
                "id": track.display_id,
                "bbox": [round(float(v), 1) for v in track.bbox],
                "state": "RECOVERING" if track.state == "RECOVERING" else "ACTIVE",
                "confidence": round(float(track.confidence), 3),
                "missing": 0,
                "recovered": track.recovered,
                "hits": track.hits,
            })
        return result

    # -- convenience -------------------------------------------------------------
    def detect(self, frame) -> List[Detection]:
        if self.detector is None:
            raise RuntimeError("TrackManager has no detector attached")
        return self.detector.detect(frame)
