"""ARGUS-X web server: job runner + dashboard API (standard library only)."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from core.benchmark import run_benchmark
from core.config import ARGUSConfig
from core.detector import YOLODetector
from core.live import LiveCamera
from core.pipeline import run_video, save_report

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")
JOBS_DIR = os.path.join(ROOT, "jobs")
UPLOAD_DIR = os.path.join(JOBS_DIR, "uploads")
DEMO_VIDEO = os.path.join(ROOT, "test.mp4")
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

JOB_ID_RE = re.compile(r"^[a-f0-9]{10}$")
JOBS: dict = {}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()
BENCHMARK: dict = {"state": "empty", "result": None, "error": None}
BENCH_LOCK = threading.Lock()
DETECTOR = None
DETECTOR_LOCK = threading.Lock()
CAMERA = None
CAMERA_LOCK = threading.Lock()
LATEST_FILE = os.path.join(JOBS_DIR, "latest.json")
BENCHMARK_FILE = os.path.join(JOBS_DIR, "benchmark.json")

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


def _rgb(color):
    b, g, r = color
    return f"#{r:02x}{g:02x}{b:02x}"


# --------------------------------------------------------------------------- jobs
def create_job(source: str, max_frames: int) -> dict:
    job_id = uuid.uuid4().hex[:10]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    job = {
        "id": job_id,
        "state": "queued",
        "progress": 0.0,
        "processed": 0,
        "total": max_frames,
        "identities": 0,
        "active": 0,
        "recoveries": 0,
        "error": None,
        "report": None,
        "frames": 0,
        "source": source,
        "dir": job_dir,
        "started": time.time(),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    threading.Thread(target=_run_job, args=(job, source, max_frames), daemon=True).start()
    return job


class LockedDetector:
    """Serialise inference so a camera stream and a video job can share weights."""

    def __init__(self, inner):
        self._inner = inner
        self._lock = threading.Lock()

    def detect(self, frame):
        with self._lock:
            return self._inner.detect(frame)

    def load(self):
        with self._lock:
            return self._inner.load()


def _shared_detector(config):
    """Keep the YOLO weights loaded between jobs (saves several seconds)."""
    global DETECTOR
    with DETECTOR_LOCK:
        if DETECTOR is None:
            detector = YOLODetector(config)
            detector.load()
            DETECTOR = LockedDetector(detector)
        return DETECTOR


def _progress(job: dict, payload: dict) -> None:
    job.update(
        {
            "state": "running",
            "progress": round(float(payload.get("fraction", 0.0)), 4),
            "processed": int(payload.get("processed", 0)),
            "total": int(payload.get("total", job["total"]) or job["total"]),
            "identities": int(payload.get("identities", 0)),
            "active": int(payload.get("active", 0)),
            "recoveries": int(payload.get("recoveries", 0)),
        }
    )


def _run_job(job: dict, source: str, max_frames: int) -> None:
    acquired = RUN_LOCK.acquire(timeout=1.0)
    try:
        job["state"] = "running"
        frames_dir = os.path.join(job["dir"], "frames")
        out_video = os.path.join(job["dir"], "argus_x.mp4")

        config = ARGUSConfig()
        report = run_video(
            source,
            config=config,
            detector=_shared_detector(config),
            out_video=out_video,
            frame_dir=frames_dir,
            max_frames=max_frames,
            progress=lambda payload: _progress(job, payload),
        )
        report["job_id"] = job["id"]
        report["available_frames"] = len(os.listdir(frames_dir))
        save_report(report, os.path.join(job["dir"], "report.json"))
        try:
            with open(LATEST_FILE, "w", encoding="utf-8") as handle:
                json.dump({"job_id": job["id"]}, handle)
        except OSError:
            pass
        job["report"] = report
        job["frames"] = report["available_frames"]
        job["progress"] = 1.0
        job["identities"] = report["summary"]["identities"]
        job["recoveries"] = report["summary"]["recoveries"]
        job["state"] = "done"
    except Exception as exc:  # surfaced in the UI, never swallowed
        job["state"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if acquired:
            RUN_LOCK.release()


def restore_benchmark() -> None:
    try:
        with open(BENCHMARK_FILE, "r", encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return
    with BENCH_LOCK:
        BENCHMARK["result"] = result
        BENCHMARK["state"] = "done"


def restore_last_job() -> None:
    """Reload the most recent finished job so a refresh keeps its results."""
    try:
        with open(LATEST_FILE, "r", encoding="utf-8") as handle:
            job_id = json.load(handle).get("job_id")
    except (OSError, json.JSONDecodeError):
        return
    if not job_id or not JOB_ID_RE.match(job_id):
        return
    job_dir = os.path.join(JOBS_DIR, job_id)
    report_path = os.path.join(job_dir, "report.json")
    frames_dir = os.path.join(job_dir, "frames")
    if not os.path.isfile(report_path) or not os.path.isdir(frames_dir):
        return
    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return
    frames = len([n for n in os.listdir(frames_dir) if n.endswith(".jpg")])
    if frames == 0:
        return
    JOBS[job_id] = {
        "id": job_id,
        "state": "done",
        "progress": 1.0,
        "processed": report.get("video", {}).get("frames_processed", frames),
        "total": frames,
        "identities": report.get("summary", {}).get("identities", 0),
        "active": report.get("summary", {}).get("active", 0),
        "recoveries": report.get("summary", {}).get("recoveries", 0),
        "error": None,
        "report": report,
        "frames": frames,
        "source": "restored",
        "dir": job_dir,
        "started": time.time(),
    }


def _benchmark_worker() -> None:
    with BENCH_LOCK:
        BENCHMARK["state"] = "running"
        BENCHMARK["error"] = None
    try:
        result = run_benchmark()
        with BENCH_LOCK:
            BENCHMARK["result"] = result
            BENCHMARK["state"] = "done"
        try:
            with open(BENCHMARK_FILE, "w", encoding="utf-8") as handle:
                json.dump(result, handle)
        except OSError:
            pass
    except Exception as exc:
        with BENCH_LOCK:
            BENCHMARK["state"] = "error"
            BENCHMARK["error"] = f"{type(exc).__name__}: {exc}"


def start_benchmark() -> dict:
    with BENCH_LOCK:
        if BENCHMARK["state"] in ("running", "done"):
            return dict(BENCHMARK)
        BENCHMARK["state"] = "running"
    threading.Thread(target=_benchmark_worker, daemon=True).start()
    return {"state": "running"}


# -------------------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    server_version = "ARGUS-X/1.0"

    def log_message(self, fmt, *args):
        pass

    # -- helpers ---------------------------------------------------------------
    def _json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message, status=400):
        self._json({"error": message}, status)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _static(self, relative):
        relative = relative.lstrip("/")
        path = os.path.normpath(os.path.join(WEB_DIR, relative))
        if not path.startswith(WEB_DIR) or not os.path.isfile(path):
            return False
        ext = os.path.splitext(path)[1].lower()
        with open(path, "rb") as handle:
            body = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)
        return True

    # -- GET -------------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            if not self._static("index.html"):
                self._error("dashboard missing", 500)
            return

        if path == "/api/health":
            self._json(
                {
                    "status": "online",
                    "demo": os.path.isfile(DEMO_VIDEO),
                    "jobs": len(JOBS),
                    "benchmark": BENCHMARK["state"],
                }
            )
            return

        if path == "/api/config":
            self._json(ARGUSConfig().to_dict())
            return

        if path == "/api/camera/status":
            camera = CAMERA
            self._json(camera.status() if camera is not None else {"state": "idle"})
            return

        if path == "/api/latest":
            done = [j for j in JOBS.values() if j["state"] == "done"]
            done.sort(key=lambda j: j.get("started", 0))
            self._json({"job_id": done[-1]["id"] if done else None})
            return

        if path == "/api/benchmark":
            if query.get("run") == ["1"]:
                self._json(start_benchmark())
                return
            with BENCH_LOCK:
                self._json(dict(BENCHMARK))
            return

        if path.startswith("/api/job/"):
            job_id = path.rsplit("/", 1)[-1]
            job = JOBS.get(job_id)
            if not job:
                self._error("unknown job", 404)
                return
            payload = {
                "id": job["id"],
                "state": job["state"],
                "progress": job["progress"],
                "processed": job["processed"],
                "total": job["total"],
                "identities": job["identities"],
                "active": job["active"],
                "recoveries": job["recoveries"],
                "frames": job["frames"],
                "error": job["error"],
            }
            if job["state"] == "done":
                payload["report"] = job["report"]
            self._json(payload)
            return

        if path == "/media/camera/stream.mjpg":
            self._camera_stream()
            return

        if path.startswith("/media/"):
            parts = path.strip("/").split("/")
            if len(parts) >= 3 and JOB_ID_RE.match(parts[1]):
                job = JOBS.get(parts[1])
                if not job:
                    self._error("unknown job", 404)
                    return
                if parts[2] == "stream.mjpg":
                    self._stream(job)
                    return
                if parts[2] == "download.mp4":
                    self._file(os.path.join(job["dir"], "argus_x.mp4"), download=True)
                    return
                if parts[2] == "frame" and len(parts) >= 4:
                    name = os.path.basename(parts[3])
                    if not name.endswith(".jpg"):
                        self._error("bad frame", 400)
                        return
                    self._file(os.path.join(job["dir"], "frames", name))
                    return
            self._error("not found", 404)
            return

        if not self._static(path):
            self._error("not found", 404)

    # -- POST ------------------------------------------------------------------
    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/run":
            try:
                payload = json.loads(self._read_body() or b"{}")
            except json.JSONDecodeError:
                self._error("invalid JSON")
                return

            max_frames = int(payload.get("max_frames", 240) or 240)
            max_frames = max(20, min(max_frames, 2000))
            source_kind = payload.get("source", "demo")

            if source_kind == "upload":
                upload_id = str(payload.get("upload_id", ""))
                if not re.match(r"^[a-f0-9]{10}$", upload_id):
                    self._error("invalid upload id")
                    return
                source = os.path.join(UPLOAD_DIR, f"{upload_id}.mp4")
                if not os.path.isfile(source):
                    self._error("upload not found", 404)
                    return
            else:
                if not os.path.isfile(DEMO_VIDEO):
                    self._error("demo video test.mp4 not found", 404)
                    return
                source = DEMO_VIDEO

            with JOBS_LOCK:
                active = [j for j in JOBS.values() if j["state"] in ("queued", "running")]
            if len(active) >= 1:
                self._json({"error": "a job is already running", "job_id": active[0]["id"]}, 409)
                return

            job = create_job(source, max_frames)
            self._json({"job_id": job["id"], "state": job["state"]})
            return

        if path == "/api/upload":
            query = parse_qs(parsed.query)
            name = (query.get("name", ["clip.mp4"])[0] or "clip.mp4")[:80]
            body = self._read_body()
            if len(body) < 2048:
                self._error("file too small")
                return
            if len(body) > 400 * 1024 * 1024:
                self._error("file larger than 400 MB")
                return
            upload_id = uuid.uuid4().hex[:10]
            path_out = os.path.join(UPLOAD_DIR, f"{upload_id}.mp4")
            with open(path_out, "wb") as handle:
                handle.write(body)
            self._json({"upload_id": upload_id, "bytes": len(body), "name": name})
            return

        if path == "/api/benchmark":
            self._json(start_benchmark())
            return

        if path == "/api/camera/start":
            global CAMERA
            with CAMERA_LOCK:
                if CAMERA is None or not CAMERA.running:
                    CAMERA = LiveCamera(
                        config=ARGUSConfig(), detector=_shared_detector(ARGUSConfig())
                    )
                    CAMERA.start()
                camera = CAMERA
            # A device can take a moment to deliver its first frame.
            deadline = time.time() + 6.0
            while camera.status()["state"] in ("starting", "idle") and time.time() < deadline:
                time.sleep(0.1)
            status = camera.status()
            self._json(status, 200 if status["state"] == "running" else 503)
            return

        if path == "/api/camera/stop":
            with CAMERA_LOCK:
                camera = CAMERA
            if camera is not None:
                camera.stop()
            self._json({"state": "idle"})
            return

        self._error("not found", 404)

    # -- media -----------------------------------------------------------------
    def _file(self, path, download=False):
        if not os.path.isfile(path):
            self._error("missing media", 404)
            return
        size = os.path.getsize(path)
        ext = os.path.splitext(path)[1].lower()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(size))
        if download:
            self.send_header("Content-Disposition", 'attachment; filename="argus_x.mp4"')
        self.end_headers()
        with open(path, "rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def _mjpeg_frame(self, data: bytes):
        self.wfile.write(b"--frame\r\n")
        self.wfile.write(b"Content-Type: image/jpeg\r\n")
        self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")

    def _camera_stream(self):
        camera = CAMERA
        if camera is None or not camera.running:
            self._error("camera is not running", 503)
            return
        deadline = time.time() + 5.0
        while camera.latest() is None and time.time() < deadline:
            time.sleep(0.05)
        frame = camera.latest()
        if frame is None:
            self._error("camera produced no frames", 503)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while camera.running and CAMERA is camera:
                data = camera.latest()
                if data:
                    self._mjpeg_frame(data)
                time.sleep(1.0 / 30.0)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _stream(self, job):
        frames_dir = os.path.join(job["dir"], "frames")
        if not os.path.isdir(frames_dir):
            self._error("no frames yet", 404)
            return
        names = sorted(name for name in os.listdir(frames_dir) if name.endswith(".jpg"))
        if not names:
            self._error("no frames yet", 404)
            return

        fps = float(job.get("report", {}).get("video", {}).get("source_fps", 12) or 12) if job.get("report") else 12.0
        delay = 1.0 / max(4.0, min(fps, 24.0))
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        index = 0
        try:
            while True:
                name = names[index % len(names)]
                try:
                    with open(os.path.join(frames_dir, name), "rb") as handle:
                        data = handle.read()
                except OSError:
                    break
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                index += 1
                time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def serve(host: str = "127.0.0.1", port: int = 8000):
    restore_last_job()
    restore_benchmark()
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    print(f"ARGUS-X dashboard -> http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nARGUS-X stopped")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARGUS-X web dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.host, args.port)
