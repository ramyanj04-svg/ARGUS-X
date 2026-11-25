"""ARGUS-X identity tracking engine."""

from .appearance import AppearanceEncoder, cosine
from .config import ARGUSConfig, DEFAULT_CONFIG
from .detector import Detection, YOLODetector
from .memory import Identity, IdentityMemory
from .motion import CameraMotionEstimator, KalmanBoxTracker, iou_batch
from .recovery import IdentityRecovery
from .reid import ReID
from .tracking import FrameResult, Track, TrackManager

__all__ = [
    "AppearanceEncoder",
    "ARGUSConfig",
    "DEFAULT_CONFIG",
    "Detection",
    "YOLODetector",
    "Identity",
    "IdentityMemory",
    "CameraMotionEstimator",
    "KalmanBoxTracker",
    "iou_batch",
    "IdentityRecovery",
    "ReID",
    "FrameResult",
    "Track",
    "TrackManager",
    "cosine",
]
