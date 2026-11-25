"""Overlay rendering: identity boxes, lost ghosts, HUD and recovery banners."""

from __future__ import annotations

import cv2
import numpy as np

PALETTE = [
    (240, 190, 60),    # cyan-ish
    (150, 220, 90),    # green
    (80, 180, 250),    # amber
    (200, 130, 250),   # violet
    (240, 140, 80),    # blue
    (200, 240, 140),   # lime
    (120, 120, 250),   # coral
    (250, 200, 160),   # periwinkle
]

RECOVERED_COLOR = (60, 220, 250)
LOST_COLOR = (150, 150, 160)
PROVISIONAL_COLOR = (170, 170, 170)


def _color_for(global_id: str):
    try:
        index = int(str(global_id).split("-")[-1])
    except (ValueError, TypeError):
        index = abs(hash(str(global_id)))
    return PALETTE[index % len(PALETTE)]


def _dashed_rect(frame, p1, p2, color, thickness=2, dash=8):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def _label(frame, text, origin, color, scale, thickness=2):
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    height, width = frame.shape[:2]
    # Keep labels inside the frame (objects at the edges used to clip them).
    x = int(min(max(4, x), max(4, width - tw - 10)))
    y = int(min(max(y, th + 6), height - 8))
    cv2.rectangle(frame, (x - 3, y - th - 5), (x + tw + 6, y + baseline + 1), (12, 14, 18), -1)
    cv2.rectangle(frame, (x - 3, y - th - 5), (x - 1, y + baseline + 1), color, -1)
    cv2.putText(frame, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245),
                thickness, cv2.LINE_AA)


def draw_frame(frame, result, identity_total: int = 0, show_hud: bool = True):
    """Render one frame's tracking state onto a copy of ``frame``."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    s = max(w / 1280.0, 0.55)
    box_thickness = max(2, int(2 * s))
    font_scale = 0.52 * s

    # lost identities: dashed ghosts with a countdown
    for ghost in result.ghosts:
        x1, y1, x2, y2 = [int(v) for v in ghost["bbox"]]
        color = RECOVERED_COLOR if ghost.get("recovered") else LOST_COLOR
        _dashed_rect(canvas, (x1, y1), (x2, y2), color, max(1, box_thickness - 1))
        _label(canvas, f"{ghost['id']} lost {ghost.get('missing', 0)}f", (x1, y1 - 4),
               LOST_COLOR, font_scale * 0.85, 1)

    # live tracks
    for box in result.boxes:
        x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
        provisional = box.get("state") == "PROVISIONAL"
        recovering = box.get("state") == "RECOVERING"
        recovered = bool(box.get("recovered"))

        if provisional:
            color = PROVISIONAL_COLOR
            _dashed_rect(canvas, (x1, y1), (x2, y2), color, max(1, box_thickness - 1), dash=5)
        else:
            color = RECOVERED_COLOR if (recovered or recovering) else _color_for(box["id"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)
            # corner accents
            length = max(8, int(16 * s))
            for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                cv2.line(canvas, (cx, cy), (cx + dx * length, cy), color, box_thickness + 1)
                cv2.line(canvas, (cx, cy), (cx, cy + dy * length), color, box_thickness + 1)

        if provisional:
            text = f"CANDIDATE {box['id']}"
        else:
            suffix = " REACQ" if recovering else (" R" if recovered else "")
            text = f"{box['id']}{suffix}  {box['confidence']:.2f}"
        _label(canvas, text, (x1, y1 - 6), color, font_scale, 2 if not provisional else 1)

    # HUD
    if show_hud:
        panel_w, panel_h = int(330 * s), int(96 * s)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (12, 12), (12 + panel_w, 12 + panel_h), (10, 12, 16), -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        cv2.rectangle(canvas, (12, 12), (12 + panel_w, 12 + panel_h), (60, 220, 250), 1)

        cv2.putText(canvas, "ARGUS-X // IDENTITY LOCK", (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (60, 220, 250), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"FRAME {result.frame_index:05d}", (24, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62 * s, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas,
                    f"tracked {result.active}   lost {len(result.ghosts)}   ids {identity_total}",
                    (24, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.44 * s, (170, 180, 190), 1, cv2.LINE_AA)

        dx, dy = result.camera_shift
        if abs(dx) + abs(dy) > 1.5:
            text = f"CAM MOTION {dx:+.1f},{dy:+.1f}"
            _label(canvas, text, (w - int(240 * s), int(38 * s)), (120, 200, 255), 0.44 * s, 1)

    # recovery banner
    recovered_events = [e for e in result.events if e.get("type") == "IDENTITY_RECOVERED"]
    if recovered_events:
        event = recovered_events[-1]
        text = f"IDENTITY RE-ACQUIRED  {event['id']}  {event['confidence']:.2f}"
        scale = 0.68 * s
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x = int((w - tw) / 2)
        y = int(h - 34 * s)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x - 16, y - th - 14), (x + tw + 16, y + 12), (10, 12, 16), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (x - 16, y - th - 14), (x + tw + 16, y + 12), RECOVERED_COLOR, 2)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, RECOVERED_COLOR, 2, cv2.LINE_AA)

    return canvas
