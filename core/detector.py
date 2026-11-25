"""Detection front-end.

The tracker never talks to YOLO directly: it consumes ``Detection`` objects.
That keeps the identity layer testable with a synthetic detector (see
``core/benchmark.py``) and swappable for any other model later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    bbox: np.ndarray
    score: float
    class_id: int = 0
    class_name: str = "person"
    embedding: Optional[np.ndarray] = None
    quality: float = 1.0

    def copy(self) -> "Detection":
        return Detection(
            bbox=np.asarray(self.bbox, dtype=float).copy(),
            score=float(self.score),
            class_id=int(self.class_id),
            class_name=self.class_name,
            embedding=None if self.embedding is None else np.asarray(self.embedding).copy(),
            quality=float(self.quality),
        )


class YOLODetector:
    def __init__(self, config):
        self.cfg = config
        self.model = None

    def load(self):
        if self.model is None:
            import torch
            from ultralytics import YOLO

            # CPU inference is thread-bound; without this YOLO leaves most
            # cores idle on Windows.
            try:
                torch.set_num_threads(min(8, max(1, os.cpu_count() or 4)))
            except Exception:
                pass
            self.model = YOLO(self.cfg.model_path)
        return self.model

    @property
    def names(self):
        model = self.load()
        return model.names

    def detect(self, frame) -> List[Detection]:
        model = self.load()
        kwargs = dict(
            source=frame,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf_thresh,
            device=self.cfg.device,
            verbose=False,
        )
        if self.cfg.classes:
            kwargs["classes"] = list(self.cfg.classes)

        results = model.predict(**kwargs)
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        detections: List[Detection] = []
        for bbox, score, class_id in zip(boxes, scores, classes):
            detections.append(
                Detection(
                    bbox=np.asarray(bbox, dtype=float),
                    score=float(score),
                    class_id=int(class_id),
                    class_name=str(model.names.get(int(class_id), "object")),
                )
            )
        return detections
