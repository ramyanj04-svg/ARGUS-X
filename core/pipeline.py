"""Video processing pipeline shared by the CLI and the web dashboard."""

from __future__ import annotations

import json
import os
import time
from typing import Callable, List, Optional

import cv2
import numpy as np

from .config import ARGUSConfig
from .detector import YOLODetector
from .tracking import TrackManager
from .visualize import draw_frame


def _resize(frame, width: int):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / float(w)
    return cv2.resize(frame, (width, max(2, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def run_video(video_path: str,
              config: ARGUSConfig = None,
              detector=None,
              out_video: Optional[str] = None,
              frame_dir: Optional[str] = None,
              max_frames: int = 0,
              progress: Optional[Callable[[dict], None]] = None,
              save_frames: bool = True) -> dict:
    """Track every frame of a video and return a full JSON-able report."""
    cfg = config or ARGUSConfig()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = max_frames if max_frames and max_frames > 0 else total or 10 ** 9

    if frame_dir:
        os.makedirs(frame_dir, exist_ok=True)

    det = detector or YOLODetector(cfg)
    manager = TrackManager(cfg, det)

    writer = None
    frames_written = 0
    activity: List[dict] = []
    started = time.time()
    size = (0, 0)
    processed = 0
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)]

    try:
        while processed < limit:
            ok, frame = cap.read()
            if not ok:
                break

            frame = _resize(frame, cfg.output_width)
            size = (frame.shape[1], frame.shape[0])

            detections = det.detect(frame)
            result = manager.step(frame, detections)

            annotated = draw_frame(frame, result, len(manager.memory.identities))

            if out_video:
                if writer is None:
                    os.makedirs(os.path.dirname(os.path.abspath(out_video)), exist_ok=True)
                    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             source_fps, (annotated.shape[1], annotated.shape[0]))
                writer.write(annotated)

            if save_frames and frame_dir:
                path = os.path.join(frame_dir, f"{processed + 1:05d}.jpg")
                cv2.imwrite(path, annotated, jpeg_params)
                frames_written += 1

            processed += 1
            if processed % 2 == 0 or processed == 1:
                activity.append({
                    "frame": processed,
                    "active": result.active,
                    "lost": len(result.ghosts),
                    "provisional": result.provisional,
                    "detections": result.detections,
                    "ids": len(manager.memory.identities),
                })

            if progress and (processed % 5 == 0 or processed == 1):
                progress({
                    "processed": processed,
                    "total": int(limit),
                    "fraction": min(1.0, processed / float(limit or 1)),
                    "identities": len(manager.memory.identities),
                    "active": result.active,
                    "recoveries": sum(i.recovery_count for i in manager.memory.identities.values()),
                })
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = max(time.time() - started, 1e-6)
    summary = manager.memory.summary()
    events = manager.memory.events

    recoveries = [e for e in events if e["type"] == "IDENTITY_RECOVERED"]
    rejections = [e for e in events if e["type"] == "RECOVERY_REJECTED"]

    report = {
        "video": {
            "path": os.path.basename(video_path),
            "source_fps": round(float(source_fps), 2),
            "width": int(size[0]),
            "height": int(size[1]),
            "frames_available": int(total),
            "frames_processed": int(processed),
        },
        "runtime": {
            "seconds": round(elapsed, 2),
            "processing_fps": round(processed / elapsed, 2),
            "frames_saved": frames_written,
        },
        "config": cfg.to_dict(),
        "summary": {
            **summary,
            "detection_events": manager.stats["detections"],
            "camera_motion_frames": manager.stats["camera_frames"],
            "recovery_attempts": len(recoveries) + len(rejections),
            "rejected_recoveries": len(rejections),
            "duplicates_suppressed": manager.stats["duplicate_suppressed"],
            "recovery_precision": round(len(recoveries) / max(len(recoveries) + len(rejections), 1), 3),
        },
        "events": events[-400:],
        "timeline": manager.memory.spans(),
        "activity": activity,
        "note": (
            "ARGUS-X confidence is a fused score (detector confidence, appearance "
            "consistency, track freshness), not a ground-truth probability."
        ),
    }
    return report


def save_report(report: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return path
