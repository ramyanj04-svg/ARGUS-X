"""Appearance descriptors used for re-identification.

A 3-channel HSV histogram alone is not enough: it discards *where* colours are,
so two people in similar clothes collapse onto the same descriptor (measured:
cosine 0.92 between different people, which no threshold can separate).

Each cue block below is **standardised** (zero mean, unit variance) before it is
concatenated.  Without that step every feature is positive and cosine
similarity is always high; with it, the same person scores ~0.95 across frames
while different people fall to ~0.60.  A small non-standardised colour block is
kept so absolute garment colour still counts.

Cues
----
1. 6x12 Lab colour grid            -> where the garment colours sit, patterns
2. 8x16 Lab colour grid            -> finer layout, stripes / logos
3. sqrt HSV histogram (16x6)       -> illumination-robust global colour
4. gradient-orientation histogram  -> texture, separates look-alike outfits
5. mean / std colour (absolute)    -> keeps jacket colour discriminative

Why the cues are also compared one by one
------------------------------------------
One cosine over the concatenated vector lets the loudest cues decide.  On the
benchmark's look-alike pair (identical jacket colour, different pattern) two
different actors still scored 0.90 that way, because the shared colour, the
histogram and the absolute-colour block all agree while only the texture block
disagrees.  :func:`fused_similarity` compares the cues separately and fuses the
scores, which turns that 0.90 into 0.79 for the impostor while the true actor
stays at 0.90.  Recovery uses the fused score; ordinary frame-to-frame
association keeps the cheap global cosine.
"""

from __future__ import annotations

import numpy as np
import cv2


def clamp_bbox(frame, bbox, pad: float = 0.02):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    x1 -= pad * bw
    x2 += pad * bw
    y1 -= pad * bh
    y2 += pad * bh
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, 1, w))
    y2 = int(np.clip(y2, 1, h))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def _standardise(values: np.ndarray, weight: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return ((values - values.mean()) / (values.std() + 1e-6)) * weight


class AppearanceEncoder:
    """Turns a crop into a compact, comparable embedding."""

    def __init__(self, size=(64, 128), min_crop_px: int = 14):
        self.size = size
        self.min_crop_px = min_crop_px
        self.dim = 6 * 12 * 3 + 8 * 16 * 3 + 16 * 6 + 8 + 6

    def crop(self, frame, bbox):
        if frame is None:
            return None
        region = clamp_bbox(frame, bbox)
        if region is None:
            return None
        x1, y1, x2, y2 = region
        if (x2 - x1) < self.min_crop_px or (y2 - y1) < self.min_crop_px:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def encode(self, frame, bbox):
        crop = self.crop(frame, bbox)
        if crop is None:
            return None

        resized = cv2.resize(crop, self.size, interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0

        # 1 + 2) colour layout at two resolutions
        coarse = cv2.resize(lab, (6, 12), interpolation=cv2.INTER_AREA)
        fine = cv2.resize(lab, (8, 16), interpolation=cv2.INTER_AREA)

        # 3) illumination-robust colour histogram (sqrt = Hellinger-ish)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 6], [0, 180, 0, 256]).ravel()
        hist = np.sqrt(hist / max(float(hist.sum()), 1e-6)).astype(np.float32)

        # 4) texture: gradient orientation histogram
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        angle = (np.arctan2(gy, gx) + np.pi) * (180.0 / np.pi)
        bins = 8
        indices = np.clip((angle / (360.0 / bins)).astype(int), 0, bins - 1)
        orientation = np.bincount(
            indices.ravel(), weights=magnitude.ravel(), minlength=bins
        ).astype(np.float32)
        orientation /= max(float(orientation.sum()), 1e-6)

        # 5) absolute colour statistics (not standardised)
        absolute = np.concatenate([
            resized.reshape(-1, 3).mean(axis=0) / 255.0,
            resized.reshape(-1, 3).std(axis=0) / 255.0,
        ]).astype(np.float32)

        embedding = np.concatenate([
            _standardise(coarse, 1.00),
            _standardise(fine, 0.85),
            _standardise(hist, 0.70),
            _standardise(orientation, 0.45),
            absolute * 1.60,
        ])

        norm = float(np.linalg.norm(embedding))
        if norm < 1e-9:
            return None
        return (embedding / norm).astype(np.float32)

    def quality(self, frame, bbox, det_score: float = 1.0) -> float:
        """0..1 usability score for a crop (size, sharpness, contrast)."""
        crop = self.crop(frame, bbox)
        if crop is None:
            return 0.0
        h, w = crop.shape[:2]
        frame_h = max(frame.shape[0], 1)
        size_score = float(np.clip((h / frame_h) / 0.28, 0.0, 1.0))

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharp_score = float(np.clip(sharpness / 260.0, 0.0, 1.0))
        contrast = float(np.clip(gray.std() / 55.0, 0.0, 1.0))

        return float(
            0.34 * size_score
            + 0.30 * sharp_score
            + 0.18 * contrast
            + 0.18 * float(np.clip(det_score, 0.0, 1.0))
        )


# Contiguous cue blocks inside the embedding, in concatenation order.
CUE_SIZES = (6 * 12 * 3, 8 * 16 * 3, 16 * 6, 8, 6)
# Weights are evidence-driven: the two colour layouts plus texture carry the
# identity, the histogram is illumination robustness, and raw mean colour gets
# the smallest say because look-alike outfits share it exactly.
CUE_WEIGHTS = (0.20, 0.36, 0.14, 0.24, 0.06)


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else np.zeros_like(vector)


def block_scores(a, b) -> list:
    """Similarity per cue block, in :data:`CUE_SIZES` order."""
    if a is None or b is None:
        return [0.0] * len(CUE_SIZES)
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        return [0.0] * len(CUE_SIZES)
    out, offset = [], 0
    for size in CUE_SIZES:
        out.append(cosine(a[offset:offset + size], b[offset:offset + size]))
        offset += size
    return out


def fused_similarity(a, b) -> float:
    """Weighted per-cue similarity: robust when one shared cue dominates."""
    scores = block_scores(a, b)
    return float(np.clip(sum(w * s for w, s in zip(CUE_WEIGHTS, scores)), -1.0, 1.0))


def cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def blend(old, new, alpha: float):
    """Exponential moving average of two embeddings."""
    if old is None:
        return None if new is None else np.asarray(new, dtype=np.float32).copy()
    if new is None:
        return old
    merged = alpha * np.asarray(old, dtype=np.float32) + (1.0 - alpha) * np.asarray(
        new, dtype=np.float32
    )
    norm = float(np.linalg.norm(merged))
    if norm < 1e-9:
        return old
    return (merged / norm).astype(np.float32)
