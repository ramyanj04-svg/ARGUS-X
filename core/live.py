"""Live camera ingestion.

The dashboard's recorded-video tab reads frames from disk.  This module is the
same engine pointed at a real webcam: one capture thread, the shared detector,
a full :class:`core.tracking.TrackManager` and the same identity overlays.

Only the newest frame is kept, so slow clients can never build a backlog; the
MJPEG endpoint in ``server.py`` simply re-serves that buffer.  Stopping the
thread releases the device, which matters on Windows where a camera stays busy
until the handle is closed.
"""

from __future__ import annotations

import threading
import time

import cv2

from .config import ARGUSConfig
from .tracking import TrackManager
from .visualize import draw_frame


class LiveCamera:
    def __init__(self, config: ARGUSConfig = None, detector=None, index: int = None):
        self.cfg = config or ARGUSConfig()
        self.detector = detector
        self.index = self.cfg.live_camera_index if index is None else int(index)
        self.width = int(self.cfg.live_width)
        self.state = "idle"          # idle | starting | running | error
        self.error = None
        self.frames = 0
        self.fps = 0.0
        self.resolution = (0, 0)
        self.started = 0.0
        self._jpeg = None
        self._jpeg_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._manager = None

    # -- lifecycle ---------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.state == "running"

    def start(self) -> "LiveCamera":
        if self._thread is not None and self._thread.is_alive():
            return self
        self.state = "starting"
        self.error = None
        self.frames = 0
        self.fps = 0.0
        self.started = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="argus-camera", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=4.0)
        if self.state != "error":
            self.state = "idle"

    def latest(self):
        with self._jpeg_lock:
            return self._jpeg

    def status(self) -> dict:
        manager = self._manager
        identities = len(manager.memory.identities) if manager is not None else 0
        active = 0
        if manager is not None:
            active = sum(1 for t in manager.tracks if t.time_since_update == 0)
        return {
            "state": self.state,
            "error": self.error,
            "frames": self.frames,
            "fps": round(self.fps, 2),
            "identities": identities,
            "active": active,
            "recoveries": sum(i.recovery_count for i in manager.memory.identities.values())
            if manager is not None else 0,
            "resolution": list(self.resolution),
            "index": self.index,
        }

    # -- capture loop ------------------------------------------------------------
    def _open(self):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            capture = cv2.VideoCapture(self.index, backend)
            if capture.isOpened():
                ok, _ = capture.read()
                if ok:
                    return capture
            capture.release()
        return None

    def _loop(self) -> None:
        capture = self._open()
        if capture is None:
            self.state = "error"
            self.error = f"camera {self.index} unavailable or already in use"
            return

        self._manager = TrackManager(self.cfg, self.detector)
        self.state = "running"
        self.started = time.time()
        failures = 0
        last = time.perf_counter()
        try:
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    failures += 1
                    if failures > 40:
                        self.state = "error"
                        self.error = "camera stopped delivering frames"
                        break
                    time.sleep(0.04)
                    continue
                failures = 0

                height, width = frame.shape[:2]
                if width > self.width:
                    scale = self.width / float(width)
                    frame = cv2.resize(frame, (self.width, max(2, int(round(height * scale)))),
                                       interpolation=cv2.INTER_AREA)
                # Mirror the webcam so moving left on screen moves left in the room.
                frame = cv2.flip(frame, 1)

                detections = self.detector.detect(frame) if self.detector is not None else []
                result = self._manager.step(frame, detections)
                canvas = draw_frame(frame, result, len(self._manager.memory.identities))
                ok, buffer = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if ok:
                    with self._jpeg_lock:
                        self._jpeg = buffer.tobytes()

                self.frames += 1
                self.resolution = (int(canvas.shape[1]), int(canvas.shape[0]))
                now = time.perf_counter()
                instant = 1.0 / max(now - last, 1e-6)
                self.fps = instant if self.fps == 0.0 else 0.85 * self.fps + 0.15 * instant
                last = now
        except Exception as exc:      # never die silently: the UI shows this
            self.state = "error"
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            capture.release()
            if self.state != "error":
                self.state = "idle"
