"""Motion: Kalman filtering in image space + global camera motion estimation."""

from __future__ import annotations

import numpy as np
import cv2


# --------------------------------------------------------------------------- boxes
def xyxy_to_xyah(box) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h], dtype=float)


def xyah_to_xyxy(state) -> np.ndarray:
    cx, cy, a, h = float(state[0]), float(state[1]), float(state[2]), max(float(state[3]), 1.0)
    w = max(a * h, 1.0)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=float)


def iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two arrays of xyxy boxes."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = inter_w * inter_h

    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter
    return np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)


def center_distance(a, b) -> float:
    ax, ay = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return float(np.hypot(ax - bx, ay - by))


# --------------------------------------------------------------------------- kalman
class KalmanBoxTracker:
    """Constant-velocity Kalman filter on (cx, cy, aspect, height).

    The filter is what keeps an identity alive while the detector reports
    nothing: it predicts *where the object should be* so that, when the object
    reappears, the spatial gate and the appearance gate can both be checked.
    """

    _std_pos = 1.0 / 20.0
    _std_vel = 1.0 / 160.0

    def __init__(self, bbox):
        measurement = xyxy_to_xyah(bbox)
        h = measurement[3]

        self.mean = np.r_[measurement, 0.0, 0.0, 0.0, 0.0]
        std = [
            2 * self._std_pos * h,
            2 * self._std_pos * h,
            1e-2,
            2 * self._std_pos * h,
            10 * self._std_vel * h,
            10 * self._std_vel * h,
            1e-5,
            10 * self._std_vel * h,
        ]
        self.cov = np.diag(np.square(np.asarray(std, dtype=float)))

        self._F = np.eye(8)
        for i in range(4):
            self._F[i, i + 4] = 1.0
        self._H = np.eye(4, 8)

    # -- prediction / correction ------------------------------------------------
    def predict(self) -> np.ndarray:
        h = max(self.mean[3], 1.0)
        std_pos = [
            self._std_pos * h,
            self._std_pos * h,
            1e-2,
            self._std_pos * h,
        ]
        std_vel = [
            self._std_vel * h,
            self._std_vel * h,
            1e-5,
            self._std_vel * h,
        ]
        q = np.diag(np.square(np.asarray(std_pos + std_vel, dtype=float)))

        self.mean = self._F @ self.mean
        self.cov = self._F @ self.cov @ self._F.T + q

        # Keep the geometry physically meaningful.
        self.mean[3] = max(self.mean[3], 8.0)
        self.mean[2] = float(np.clip(self.mean[2], 0.12, 6.0))
        return self.bbox

    def update(self, bbox) -> None:
        z = xyxy_to_xyah(bbox)
        h = max(z[3], 1.0)
        std = [self._std_pos * h, self._std_pos * h, 1e-2, self._std_pos * h]
        r = np.diag(np.square(np.asarray(std, dtype=float)))

        projected_mean = self._H @ self.mean
        projected_cov = self._H @ self.cov @ self._H.T + r
        try:
            kalman_gain = self.cov @ self._H.T @ np.linalg.inv(projected_cov)
        except np.linalg.LinAlgError:
            return

        innovation = z - projected_mean
        self.mean = self.mean + kalman_gain @ innovation
        self.cov = self.cov - kalman_gain @ self._H @ self.cov

    # -- helpers ----------------------------------------------------------------
    @property
    def bbox(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.mean[4], self.mean[5]], dtype=float)

    def mahalanobis(self, bbox) -> float:
        z = xyxy_to_xyah(bbox)
        h = max(self.mean[3], 1.0)
        std = [self._std_pos * h, self._std_pos * h, 1e-2, self._std_pos * h]
        r = np.diag(np.square(np.asarray(std, dtype=float)))
        projected_mean = self._H @ self.mean
        projected_cov = self._H @ self.cov @ self._H.T + r
        try:
            return float((z - projected_mean) @ np.linalg.solve(projected_cov, z - projected_mean))
        except np.linalg.LinAlgError:
            return float("inf")

    def shift(self, dx: float, dy: float) -> None:
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return
        self.mean[0] += dx
        self.mean[1] += dy

    def damp_velocity(self, factor: float) -> None:
        self.mean[4] *= factor
        self.mean[5] *= factor


# ------------------------------------------------------------------- camera motion
class CameraMotionEstimator:
    """Estimates how far the *scene* moved between two frames.

    When the camera pans, every Kalman prediction is in the previous frame's
    coordinate system.  Without compensation, fast pans look like objects
    teleporting, association fails and identities fragment.  We estimate a
    robust translation from ORB matches on the background and shift all tracks
    into the current frame before matching.
    """

    def __init__(self, work_width: int = 320, max_shift_ratio: float = 0.20):
        self.work_width = work_width
        self.max_shift_ratio = max_shift_ratio
        self._prev_gray = None
        self._orb = cv2.ORB_create(nfeatures=500, fastThreshold=9)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last_shift = (0.0, 0.0)

    def _prepare(self, frame):
        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            return None, 1.0
        scale = self.work_width / float(w)
        small = cv2.resize(frame, (self.work_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return gray, scale

    def estimate(self, frame) -> tuple:
        gray, scale = self._prepare(frame)
        if gray is None:
            return (0.0, 0.0)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return (0.0, 0.0)

        dx = dy = 0.0
        try:
            kp_a, des_a = self._orb.detectAndCompute(self._prev_gray, None)
            kp_b, des_b = self._orb.detectAndCompute(gray, None)
            if des_a is not None and des_b is not None and len(kp_a) >= 12 and len(kp_b) >= 12:
                matches = self._matcher.match(des_a, des_b)
                if len(matches) >= 12:
                    src = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    model, inliers = cv2.estimateAffinePartial2D(
                        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
                    )
                    if model is not None and inliers is not None:
                        inlier_mask = inliers.ravel().astype(bool)
                        if inlier_mask.sum() >= 10:
                            delta = (dst[inlier_mask] - src[inlier_mask]).reshape(-1, 2)
                            dx, dy = np.median(delta, axis=0)
        except cv2.error:
            dx = dy = 0.0

        self._prev_gray = gray

        limit_x = frame.shape[1] * self.max_shift_ratio
        limit_y = frame.shape[0] * self.max_shift_ratio
        dx = float(np.clip(dx / max(scale, 1e-6), -limit_x, limit_x))
        dy = float(np.clip(dy / max(scale, 1e-6), -limit_y, limit_y))

        # Ignore jitter smaller than a pixel.
        if abs(dx) < 1.0 and abs(dy) < 1.0:
            dx = dy = 0.0

        self.last_shift = (dx, dy)
        return (dx, dy)

    def reset(self) -> None:
        self._prev_gray = None
        self.last_shift = (0.0, 0.0)
