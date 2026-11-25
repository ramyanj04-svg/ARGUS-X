"""Central configuration for the ARGUS-X tracking engine."""

from dataclasses import dataclass, asdict
from typing import Optional, Tuple


@dataclass
class ARGUSConfig:
    """All tunable knobs in one place.

    Defaults are tuned for a CPU-only demo on street / room footage with
    people.  Every stage of the pipeline reads from this object so the whole
    system can be re-configured (or ablated) from the CLI or the web API.
    """

    # ------------------------------------------------------------------ detection
    model_path: str = "yolo11n.pt"
    imgsz: int = 512
    conf_thresh: float = 0.20          # lowest score kept from the detector
    high_thresh: float = 0.45          # first-stage association detections
    new_track_thresh: float = 0.45     # score needed to spawn a brand new track
    classes: Optional[Tuple[int, ...]] = (0,)   # COCO person; None -> every class
    device: str = "cpu"

    # -------------------------------------------------------------------- motion
    iou_gate: float = 0.20             # min IoU before appearance can rescue a pair
    mahalanobis_gate: float = 13.8     # chi-square 95% for 4 DoF
    camera_compensation: bool = True
    velocity_decay: float = 0.94       # damping applied to velocity while lost

    # --------------------------------------------------------------- association
    w_iou: float = 0.45
    w_app: float = 0.55
    app_gate: float = 0.58             # below this the pair is never associated
    low_conf_app_gate: float = 0.72    # low-score detections need stronger proof
    reappear_app_gate: float = 0.70    # first frame after a gap
    max_iou_distance: float = 0.85

    # ---------------------------------------------------------------- lifecycle
    n_init: int = 3                    # hits before a tentative track is confirmed
    identity_grace: int = 40           # frames a new track may stay unidentified while
                                       # it waits for a clean view to claim a lost ID
    max_age: int = 90                  # absolute cap for a track (safety net)
    reid_after: int = 12               # after this gap a track dies and only the
                                       # verified recovery path can restore it
    archive_after: int = 900           # frames before an identity is archived
    duplicate_iou: float = 0.70        # suppression of overlapping new tracks

    # --------------------------------------------------------------- appearance
    gallery_size: int = 12
    ema_alpha: float = 0.85            # weight of the old embedding
    min_crop_px: int = 28
    gallery_quality: float = 0.28      # min quality to add a crop to the gallery

    # ------------------------------------------------------------------ recovery
    reid_threshold: float = 0.70       # base appearance threshold for a recovery
    recovery_threshold: float = 0.80   # fused multi-cue threshold for a recovery
    reid_margin: float = 0.045         # required lead over the runner-up identity
    recovery_confirm: int = 3          # frames a recovery must survive to be kept
    recovery_min_gap: int = 3          # frames an identity must be gone before reid
    search_radius_scale: float = 1.50  # spatial gate, in body heights
    search_radius_slack: int = 220     # frames before the gate stops growing

    # -------------------------------------------------------------------- live
    live_camera_index: int = 0         # webcam device index for the live tab
    live_width: int = 960              # frames are downscaled before inference

    # ------------------------------------------------------------------ runtime
    output_width: int = 1280
    jpeg_quality: int = 78
    draw_ghosts: bool = True

    def to_dict(self) -> dict:
        data = asdict(self)
        if data.get("classes") is not None:
            data["classes"] = list(data["classes"])
        return data


DEFAULT_CONFIG = ARGUSConfig()
