# app/components/gaze_overlay.py

import cv2
import numpy as np
from collections import deque

# ─── Config ───────────────────────────────────────────────────────────────────
SCALE_FACTOR  = 250    # tuning knob — increase to spread dot further
TRAIL_LENGTH  = 15     # number of past gaze points to show as trail
DOT_RADIUS    = 12     # main gaze dot size
TRAIL_COLOR   = (0, 255, 120)
DOT_COLOR     = (0, 255, 0)
CROSSHAIR_COLOR = (255, 255, 255)

# ─── Trail buffer ─────────────────────────────────────────────────────────────
_trail = deque(maxlen=TRAIL_LENGTH)


def draw_gaze_overlay(scene_frame, pitch: float, yaw: float) -> None:
    """
    Draws a gaze dot on the scene frame (DroidCam feed).

    pitch → up/down   (-1.0 to 1.0)
    yaw   → left/right (-1.0 to 1.0)

    Dot position:
        gaze_x = W/2 + (yaw   × SCALE_FACTOR)
        gaze_y = H/2 + (pitch × SCALE_FACTOR)
    """
    h, w = scene_frame.shape[:2]

    # ── Calculate gaze point ──
    gaze_x = int(w / 2 + yaw   * SCALE_FACTOR)
    gaze_y = int(h / 2 + pitch * SCALE_FACTOR)

    # ── Clamp to frame bounds ──
    gaze_x = np.clip(gaze_x, DOT_RADIUS, w - DOT_RADIUS)
    gaze_y = np.clip(gaze_y, DOT_RADIUS, h - DOT_RADIUS)

    # ── Add to trail ──
    _trail.append((gaze_x, gaze_y))

    # ── Draw trail ──
    for i in range(1, len(_trail)):
        alpha = i / len(_trail)           # fade older points
        radius = max(1, int(DOT_RADIUS * 0.3 * alpha))
        color  = tuple(int(c * alpha) for c in TRAIL_COLOR)
        cv2.circle(scene_frame, _trail[i], radius, color, -1)

    # ── Draw crosshair ──
    cross_size = DOT_RADIUS + 8
    cv2.line(scene_frame,
             (gaze_x - cross_size, gaze_y),
             (gaze_x + cross_size, gaze_y),
             CROSSHAIR_COLOR, 1)
    cv2.line(scene_frame,
             (gaze_x, gaze_y - cross_size),
             (gaze_x, gaze_y + cross_size),
             CROSSHAIR_COLOR, 1)

    # ── Draw outer ring ──
    cv2.circle(scene_frame, (gaze_x, gaze_y), DOT_RADIUS + 4, DOT_COLOR, 1)

    # ── Draw main dot ──
    cv2.circle(scene_frame, (gaze_x, gaze_y), DOT_RADIUS, DOT_COLOR, -1)

    # ── Draw center pinpoint ──
    cv2.circle(scene_frame, (gaze_x, gaze_y), 2, (0, 0, 0), -1)

    # ── Label pitch/yaw values ──
    cv2.putText(scene_frame, f"yaw: {yaw:.2f}  pitch: {pitch:.2f}",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CROSSHAIR_COLOR, 1)