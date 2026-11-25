"""Persistent identity memory.

This is the part the original ARGUS-X was missing: an identity is *not* a
tracker id.  It survives the tracker, the detector and the tracker's own
occlusion buffer.  Every identity keeps:

* an averaged appearance embedding plus a gallery of high-quality crops, so
  pose / lighting / scale drift does not break re-identification
* its last known geometry and velocity, so the search region can be predicted
* a lifecycle state and a full presence timeline (rendered by the dashboard)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .appearance import blend, cosine


@dataclass
class Identity:
    global_id: str
    class_name: str
    created_frame: int
    last_seen_frame: int
    state: str = "ACTIVE"                       # ACTIVE | LOST | ARCHIVED
    bbox: Optional[list] = None
    velocity: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    gallery: List[np.ndarray] = field(default_factory=list)
    quality: float = 1.0
    hits: int = 1
    recovery_count: int = 0
    lost_count: int = 0
    best_confidence: float = 0.0
    history: List[tuple] = field(default_factory=list)   # (frame, x1, y1, x2, y2)

    # -- geometry -----------------------------------------------------------------
    @property
    def center(self):
        if not self.bbox:
            return None
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def size(self):
        if not self.bbox:
            return (100.0, 200.0)
        x1, y1, x2, y2 = self.bbox
        return (max(x2 - x1, 1.0), max(y2 - y1, 1.0))

    def compare(self, embedding) -> float:
        """Best similarity against the gallery, with the EMA as a soft prior."""
        if embedding is None:
            return 0.0
        scores = [cosine(embedding, crop) for crop in self.gallery]
        if self.embedding is not None:
            scores.append(cosine(embedding, self.embedding) * 0.97)
        return float(max(scores)) if scores else 0.0

    def snapshot(self) -> dict:
        x1, y1, x2, y2 = self.bbox if self.bbox else (0, 0, 0, 0)
        return {
            "id": self.global_id,
            "class": self.class_name,
            "state": self.state,
            "first_frame": self.created_frame,
            "last_frame": self.last_seen_frame,
            "hits": self.hits,
            "recoveries": self.recovery_count,
            "quality": round(float(self.quality), 3),
            "gallery": len(self.gallery),
            "confidence": round(float(self.best_confidence), 3),
            "bbox": [round(float(v), 1) for v in (x1, y1, x2, y2)],
        }


class IdentityMemory:
    """Registry of every identity ever created in a session."""

    def __init__(self, gallery_size: int = 12, archive_after: int = 900):
        self.identities: Dict[str, Identity] = {}
        self.gallery_size = gallery_size
        self.archive_after = archive_after
        self.events: List[dict] = []
        self._counter = 0

    # -- creation -----------------------------------------------------------------
    def create(self, bbox, embedding, class_name: str, frame_idx: int,
               quality: float = 1.0) -> Identity:
        self._counter += 1
        global_id = f"AX-{self._counter:03d}"
        identity = Identity(
            global_id=global_id,
            class_name=class_name,
            created_frame=frame_idx,
            last_seen_frame=frame_idx,
            bbox=[float(v) for v in bbox],
            embedding=embedding,
            quality=float(quality),
            hits=1,
            best_confidence=float(quality),
        )
        if embedding is not None:
            identity.gallery.append(np.asarray(embedding, dtype=np.float32))
        self.identities[global_id] = identity
        self._log(frame_idx, "IDENTITY_CREATED", global_id, quality, "new track confirmed")
        return identity

    # -- updates ------------------------------------------------------------------
    def observe(self, global_id: str, bbox, embedding, frame_idx: int,
                quality: float = 1.0, update_appearance: bool = True) -> bool:
        identity = self.identities.get(global_id)
        if identity is None:
            return False

        old_center = identity.center
        steps = max(frame_idx - identity.last_seen_frame, 1)
        new_bbox = [float(v) for v in bbox]
        new_center = ((new_bbox[0] + new_bbox[2]) / 2.0, (new_bbox[1] + new_bbox[3]) / 2.0)

        if old_center is not None and steps <= 30:
            instant = np.array([(new_center[0] - old_center[0]) / steps,
                                (new_center[1] - old_center[1]) / steps])
            if identity.velocity is None:
                identity.velocity = instant
            else:
                identity.velocity = 0.65 * identity.velocity + 0.35 * instant

        identity.bbox = new_bbox
        identity.last_seen_frame = frame_idx
        identity.state = "ACTIVE"
        identity.hits += 1
        identity.quality = 0.88 * identity.quality + 0.12 * float(quality)
        identity.best_confidence = max(identity.best_confidence * 0.997, float(quality))
        identity.history.append((frame_idx, *new_bbox))
        if len(identity.history) > 6000:
            del identity.history[:2000]

        if embedding is not None and update_appearance:
            identity.embedding = blend(identity.embedding, embedding, 0.85)
            if quality >= 0.25:
                identity.gallery.append(np.asarray(embedding, dtype=np.float32))
                if len(identity.gallery) > self.gallery_size:
                    del identity.gallery[0]
        return True

    def mark_lost(self, global_id: str, frame_idx: int, reason: str = "no detection") -> None:
        identity = self.identities.get(global_id)
        if identity is None or identity.state == "LOST":
            return
        identity.state = "LOST"
        identity.lost_count += 1
        self._log(frame_idx, "IDENTITY_LOST", global_id, 0.0, reason)

    def mark_recovered(self, global_id: str, frame_idx: int, confidence: float,
                       detail: str = "") -> None:
        identity = self.identities.get(global_id)
        if identity is None:
            return
        identity.state = "ACTIVE"
        identity.recovery_count += 1
        identity.last_seen_frame = max(identity.last_seen_frame, frame_idx)
        self._log(frame_idx, "IDENTITY_RECOVERED", global_id, confidence, detail)

    def archive(self, global_id: str, frame_idx: int) -> None:
        identity = self.identities.get(global_id)
        if identity is None or identity.state == "ARCHIVED":
            return
        identity.state = "ARCHIVED"
        self._log(frame_idx, "IDENTITY_ARCHIVED", global_id, 0.0,
                  f"unseen for {frame_idx - identity.last_seen_frame} frames")

    def rollback(self, global_id: str, frame_idx: int, confidence: float) -> None:
        """A provisional recovery failed verification -> identity goes back to LOST."""
        identity = self.identities.get(global_id)
        if identity is None:
            return
        identity.state = "LOST"
        self._log(frame_idx, "RECOVERY_REJECTED", global_id, confidence,
                  "lost again before verification completed")

    # -- queries ------------------------------------------------------------------
    def get(self, global_id: str) -> Optional[Identity]:
        return self.identities.get(global_id)

    def known(self, exclude_ids=()) -> List[Identity]:
        """Every identity holding appearance evidence, in any lifecycle state.

        Recovery needs this list, not just the lost ones: a look-alike that is
        being tracked *right now* is the most dangerous rival for an old
        identity, and it must be able to veto a recovery.
        """
        excluded = set(exclude_ids)
        return [
            identity
            for identity in self.identities.values()
            if identity.embedding is not None and identity.global_id not in excluded
        ]

    def get_lost(self, frame_idx: int, min_gap: int = 1) -> List[Identity]:
        return [
            identity
            for identity in self.identities.values()
            if identity.state in ("LOST", "ARCHIVED")
            and (frame_idx - identity.last_seen_frame) >= min_gap
            and identity.embedding is not None
        ]

    def expire(self, frame_idx: int) -> List[str]:
        archived = []
        for identity in self.identities.values():
            if identity.state == "LOST" and (frame_idx - identity.last_seen_frame) > self.archive_after:
                identity.state = "ARCHIVED"
                self._log(frame_idx, "IDENTITY_ARCHIVED", identity.global_id, 0.0,
                          "recovery window closed")
                archived.append(identity.global_id)
        return archived

    def _log(self, frame: int, kind: str, global_id: str, confidence: float,
             detail: str = "") -> None:
        self.events.append(
            {
                "frame": int(frame),
                "type": kind,
                "id": global_id,
                "confidence": round(float(confidence), 4),
                "detail": detail,
            }
        )

    # -- reporting ----------------------------------------------------------------
    def spans(self) -> List[dict]:
        out = []
        for identity in sorted(self.identities.values(), key=lambda i: i.created_frame):
            segments = []
            for entry in identity.history:
                frame = entry[0]
                if not segments or frame - segments[-1][1] > 2:
                    segments.append([frame, frame])
                else:
                    segments[-1][1] = frame
            out.append({
                "id": identity.global_id,
                "class": identity.class_name,
                "created": identity.created_frame,
                "last": identity.last_seen_frame,
                "recoveries": identity.recovery_count,
                "state": identity.state,
                "segments": segments,
            })
        return out

    def summary(self) -> dict:
        identities = list(self.identities.values())
        lengths = [i.hits for i in identities]
        gaps = []
        for identity in identities:
            frames = [entry[0] for entry in identity.history]
            for previous, current in zip(frames, frames[1:]):
                if current - previous > 10:
                    gaps.append(current - previous)
        return {
            "identities": len(identities),
            "active": sum(1 for i in identities if i.state == "ACTIVE"),
            "lost": sum(1 for i in identities if i.state == "LOST"),
            "archived": sum(1 for i in identities if i.state == "ARCHIVED"),
            "recoveries": sum(i.recovery_count for i in identities),
            "recovered_identities": sum(1 for i in identities if i.recovery_count > 0),
            "avg_track_length": round(float(np.mean(lengths)), 1) if lengths else 0.0,
            "max_occlusion_frames": int(max(gaps)) if gaps else 0,
            "identities_list": [i.snapshot() for i in sorted(identities, key=lambda x: x.created_frame)],
        }
