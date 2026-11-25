"""Synthetic long-occlusion benchmark with exact ground truth.

The real world gives us no labels, and a hackathon demo needs numbers that can
be defended.  This module renders a controllable scene -- moving camera,
occluding pillar, look-alike people, detector noise and dropouts -- and scores
ARGUS-X against CLEAR-MOT / IDF1 using the labels it generated itself.

It also runs a plain IoU tracker (no appearance, no identity memory) on exactly
the same detections, so the comparison isolates the identity layer.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .config import ARGUSConfig
from .detector import Detection
from .metrics import mot_metrics
from .tracking import TrackManager

WIDTH, HEIGHT = 720, 400
FRAMES = 300


class SyntheticDetector:
    """Deterministic-ish noisy detector that returns the provided boxes."""

    def __init__(self, rng: np.random.Generator, drop: float = 0.05,
                 jitter: float = 2.5, false_positives: float = 0.25):
        self.rng = rng
        self.drop = drop
        self.jitter = jitter
        self.false_positives = false_positives
        self.pending: List[tuple] = []
        self.detections = 0

    def set_scene(self, boxes):
        self.pending = list(boxes)

    def detect(self, frame) -> List[Detection]:
        out = []
        for bbox in self.pending:
            if self.rng.random() < self.drop:
                continue
            x1, y1, x2, y2 = bbox
            noise = self.rng.normal(0, self.jitter, 4)
            out.append(Detection(
                bbox=np.array([x1 + noise[0], y1 + noise[1], x2 + noise[2], y2 + noise[3]]),
                score=float(np.clip(self.rng.normal(0.90, 0.035), 0.3, 0.99)),
                class_id=0,
                class_name="person",
            ))
        if self.rng.random() < self.false_positives:
            cx = float(self.rng.uniform(30, WIDTH - 60))
            cy = float(self.rng.uniform(40, HEIGHT - 60))
            out.append(Detection(
                bbox=np.array([cx, cy, cx + 26, cy + 54]),
                score=float(self.rng.uniform(0.30, 0.48)),
                class_id=0,
                class_name="person",
            ))
        self.detections += len(out)
        return out


def _draw_person(frame, cx, top, w, h, torso, pattern, accent):
    x1 = int(cx - w / 2)
    y1 = int(top)
    x2 = int(cx + w / 2)
    y2 = int(top + h)
    head_r = max(6, int(w * 0.26))
    head_cy = y1 + head_r

    cv2.circle(frame, (int(cx), head_cy), head_r, (170, 190, 215), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, head_cy + head_r), (x2, y2), torso, -1)
    # legs
    leg_h = int(h * 0.32)
    cv2.rectangle(frame, (x1 + 2, y2 - leg_h), (int(cx) - 1, y2), (55, 60, 70), -1)
    cv2.rectangle(frame, (int(cx) + 1, y2 - leg_h), (x2 - 2, y2), (55, 60, 70), -1)
    # pattern / accessory: the only cue that separates look-alike people
    if pattern == "stripe":
        for y in range(head_cy + head_r + 4, y2 - leg_h - 2, 10):
            cv2.line(frame, (x1 + 2, y), (x2 - 2, y), accent, 3)
    elif pattern == "vest":
        cv2.rectangle(frame, (x1 + 3, head_cy + head_r + 3), (x2 - 3, y2 - leg_h - 3), accent, 2)
    elif pattern == "backpack":
        cv2.rectangle(frame, (x2 - 8, head_cy + head_r + 2), (x2 + 4, y2 - leg_h - 2), accent, -1)


def _render(frame_idx: int, people, camera_dx: float, occluder_x: float):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :] = (28, 30, 36)

    # parallax background: vertical gradient + drifting grid (ORB features)
    for y in range(HEIGHT):
        shade = int(26 + 26 * (y / HEIGHT))
        frame[y, :] = (shade, shade + 2, shade + 6)
    offset = int(camera_dx * 0.55) % 80
    for gx in range(-80, WIDTH + 80, 80):
        x = gx - offset
        cv2.line(frame, (x, 0), (x, HEIGHT), (46, 50, 58), 1)
    for gy in range(0, HEIGHT, 80):
        cv2.line(frame, (0, gy), (WIDTH, gy), (46, 50, 58), 1)
    cv2.line(frame, (0, int(HEIGHT * 0.82)), (WIDTH, int(HEIGHT * 0.82)), (60, 64, 74), 2)

    boxes = []
    for person in people:
        cx = person["x"] - camera_dx
        _draw_person(frame, cx, person["top"], person["w"], person["h"],
                     person["torso"], person["pattern"], person["accent"])
        boxes.append([cx - person["w"] / 2, person["top"],
                      cx + person["w"] / 2, person["top"] + person["h"]])

    # occluding pillar drawn in front of every lane
    px1 = int(occluder_x - camera_dx)
    px2 = px1 + 120
    cv2.rectangle(frame, (px1, 0), (px2, HEIGHT), (18, 19, 24), -1)
    cv2.rectangle(frame, (px1 + 8, 0), (px2 - 8, HEIGHT), (30, 31, 37), -1)
    cv2.putText(frame, "STRUCTURE", (px1 + 16, HEIGHT - 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (70, 72, 82), 1, cv2.LINE_AA)

    visible = []
    for person, box in zip(people, boxes):
        hidden = max(0.0, min(px2, box[2]) - max(px1, box[0]))
        visible.append(hidden < 0.55 * (box[2] - box[0]))

    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    return frame, boxes, visible


def build_scene(seed: int = 7):
    """Create the scripted scenario: pan, long occlusion, look-alikes."""
    people = [
        # Two look-alike jackets in the same colour, different pattern and height.
        # Actors walk in distinct depth lanes so that the benchmark measures
        # occlusion / camera / dropout recovery rather than crowd crossing.
        {"id": "G1", "x": 60, "top": 232, "w": 36, "h": 92, "vx": 1.5,
         "torso": (58, 78, 168), "pattern": "stripe", "accent": (232, 238, 244)},
        {"id": "G2", "x": 560, "top": 300, "w": 34, "h": 78, "vx": -1.0,
         "torso": (58, 78, 168), "pattern": "solid", "accent": (0, 0, 0)},
        {"id": "G3", "x": 120, "top": 96, "w": 32, "h": 82, "vx": 0.7,
         "torso": (70, 150, 80), "pattern": "vest", "accent": (230, 190, 60)},
        {"id": "G4", "x": 640, "top": 166, "w": 38, "h": 88, "vx": -1.2,
         "torso": (150, 90, 190), "pattern": "backpack", "accent": (60, 60, 60)},
    ]

    frames, gt = [], {}
    for index in range(FRAMES):
        camera_dx = 22.0 * math.sin(2 * math.pi * index / 150.0)
        for person in people:
            person["x"] += person["vx"]
            if person["x"] < 30:
                person["x"] = 30.0
                person["vx"] = abs(person["vx"])
            elif person["x"] > WIDTH - 30:
                person["x"] = WIDTH - 30.0
                person["vx"] = -abs(person["vx"])
            person["top"] = person["top"] + 1.4 * math.sin(index / 11.0 + person["x"])

        frame, boxes, visible = _render(index + 1, people, camera_dx, occluder_x=350)
        frames.append(frame)
        gt[index + 1] = [
            (person["id"], [int(round(v)) for v in box], bool(vis))
            for person, box, vis in zip(people, boxes, visible)
        ]

    return frames, gt


class NaiveIoUTracker:
    """Baseline: greedy IoU association, no appearance, no identity memory."""

    def __init__(self, iou_threshold: float = 0.30, max_missed: int = 2):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: List[dict] = []
        self._counter = 0

    def update(self, detections: List[Detection], frame_idx: int) -> List[tuple]:
        from .motion import iou_batch

        for track in self.tracks:
            track["missed"] += 1

        out = []
        matched = []
        if self.tracks and detections:
            boxes = np.array([t["bbox"] for t in self.tracks], dtype=float)
            dets = np.array([d.bbox for d in detections], dtype=float)
            ious = iou_batch(boxes, dets)
            used_t, used_d = set(), set()
            for index in np.argsort(ious.ravel())[::-1]:
                t, d = divmod(int(index), ious.shape[1])
                if ious[t, d] < self.iou_threshold:
                    break
                if t in used_t or d in used_d:
                    continue
                used_t.add(t)
                used_d.add(d)
                matched.append((t, d))

        used_dets = set()
        for t, d in matched:
            track = self.tracks[t]
            track["bbox"] = detections[d].bbox
            track["missed"] = 0
            used_dets.add(d)
            out.append((track["id"], track["bbox"]))

        for d, det in enumerate(detections):
            if d in used_dets:
                continue
            self._counter += 1
            track = {"id": f"BASE{self._counter}", "bbox": det.bbox, "missed": 0}
            self.tracks.append(track)
            out.append((track["id"], det.bbox))

        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missed]
        return out


def _run_argus(frames, det_boxes, config: ARGUSConfig, seed: int):
    rng = np.random.default_rng(seed)
    detector = SyntheticDetector(rng)
    manager = TrackManager(config, detector)
    pred: Dict[int, List[tuple]] = {}
    for index, frame in enumerate(frames):
        frame_idx = index + 1
        detector.set_scene(det_boxes[frame_idx])
        detections = detector.detect(frame)
        result = manager.step(frame, detections)
        pred[frame_idx] = [
            (box["id"], box["bbox"]) for box in result.boxes if box["state"] != "PROVISIONAL"
        ]
    return pred, manager


def _run_baseline(frames, det_boxes, seed: int):
    rng = np.random.default_rng(seed)
    detector = SyntheticDetector(rng)
    tracker = NaiveIoUTracker()
    pred: Dict[int, List[tuple]] = {}
    for index, frame in enumerate(frames):
        frame_idx = index + 1
        detector.set_scene(det_boxes[frame_idx])
        detections = detector.detect(frame)
        pred[frame_idx] = tracker.update(detections, frame_idx)
    return pred


def run_benchmark(frames: int = None, seed: int = 7) -> dict:
    frames_all, gt = build_scene(seed=seed)
    if frames is not None:
        frames_all = frames_all[:frames]
        gt = {k: v for k, v in gt.items() if k <= frames}

    # Fully occluded objects produce no detection at all -- that is the whole
    # point of the sequence.
    det_boxes = {
        frame: [box for _, box, visible in items if visible]
        for frame, items in gt.items()
    }

    config = ARGUSConfig()
    argus_pred, manager = _run_argus(frames_all, det_boxes, config, seed)
    base_pred = _run_baseline(frames_all, det_boxes, seed)

    argus_metrics = mot_metrics(gt, argus_pred)
    base_metrics = mot_metrics(gt, base_pred)

    summary = manager.memory.summary()
    return {
        "sequence": {
            "name": "ARGUS-X synthetic long-occlusion set",
            "frames": len(frames_all),
            "resolution": f"{WIDTH}x{HEIGHT}",
            "objects": 4,
            "stressors": [
                "moving camera (sinusoidal pan, 22 px amplitude)",
                "static pillar causing 60-100 frame full occlusions",
                "two look-alike jackets in identical colour",
                "5% detector dropout, 2.5 px box jitter",
                "random low-score false positives each frame",
            ],
        },
        "argus": {**argus_metrics, "identities_created": summary["identities"],
                  "recoveries": summary["recoveries"],
                  "max_occlusion_frames": summary["max_occlusion_frames"]},
        "baseline": base_metrics,
        "ground_truth_ids": 4,
    }
