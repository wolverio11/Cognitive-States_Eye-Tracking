# app/components/mandala.py

import cv2
import numpy as np
import math
import time

# ─── Colors (BGR) ─────────────────────────────────────────────────────────────
STATE_COLORS = {
    "Vikshipta": (0, 0, 255),
    "Kshipta":   (0, 140, 255),
    "Mudha":     (0, 255, 255),
    "Ekagra":    (0, 255, 120),
    "Nirodha":   (255, 255, 255),
}


def _lerp_color(c1, c2, t):
    """Linearly interpolate between two BGR colors."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_mandala(frame, score: float, state_name: str) -> None:
    """
    Draws a sacred geometry mandala in the bottom-right corner.

    Behavior based on score:
      - Vikshipta : small, dim, slow rotation, few petals
      - Kshipta   : slightly bigger, orange glow
      - Mudha     : medium, yellow, steady
      - Ekagra    : larger, green, faster spin, more layers
      - Nirodha   : full bloom, white glow, all layers visible
    """
    h, w   = frame.shape[:2]
    t      = time.time()
    color  = STATE_COLORS.get(state_name, (255, 255, 255))

    # ── Dynamic properties based on score ──
    radius      = int(40 + score * 80)          # 40px → 120px
    num_petals  = int(4 + score * 8)            # 4 → 12 petals
    num_layers  = int(1 + score * 4)            # 1 → 5 layers
    rotation    = (t * (0.3 + score * 1.2)) % (2 * math.pi)   # slow → fast
    alpha       = 0.35 + score * 0.55          # dim → bright (overlay alpha)
    glow        = score > 0.75                  # glow effect for Ekagra+

    # ── Position: bottom-right corner ──
    cx = w - radius - 20
    cy = h - radius - 55   # above score bar

    # ── Draw on overlay for alpha blending ──
    overlay = frame.copy()

    # ── Layer loop ──
    for layer in range(num_layers):
        layer_r     = int(radius * (0.35 + layer * 0.18))
        layer_alpha = 1.0 - (layer * 0.15)
        layer_color = _lerp_color(color, (20, 20, 20), layer * 0.1)

        # Petals
        for p in range(num_petals):
            angle     = rotation + (2 * math.pi * p / num_petals)
            petal_len = layer_r
            petal_w   = max(1, int(layer_r * 0.25))

            x1 = int(cx + math.cos(angle) * petal_len * 0.4)
            y1 = int(cy + math.sin(angle) * petal_len * 0.4)
            x2 = int(cx + math.cos(angle) * petal_len)
            y2 = int(cy + math.sin(angle) * petal_len)

            cv2.line(overlay, (x1, y1), (x2, y2), layer_color, petal_w)

            # Petal tip dot
            cv2.circle(overlay, (x2, y2), max(1, petal_w // 2), layer_color, -1)

        # Inner polygon (connecting petal bases)
        pts = []
        for p in range(num_petals):
            angle = rotation + (2 * math.pi * p / num_petals)
            x     = int(cx + math.cos(angle) * layer_r * 0.4)
            y     = int(cy + math.sin(angle) * layer_r * 0.4)
            pts.append([x, y])
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], True, layer_color, 1)

        # Outer ring
        cv2.circle(overlay, (cx, cy), layer_r, layer_color, 1)

    # ── Center dot ──
    center_r = max(3, int(radius * 0.08))
    cv2.circle(overlay, (cx, cy), center_r, color, -1)

    # ── Glow effect for high states ──
    if glow:
        glow_r = int(radius * 1.1)
        glow_color = _lerp_color(color, (0, 0, 0), 0.5)
        cv2.circle(overlay, (cx, cy), glow_r, glow_color, 2)
        cv2.circle(overlay, (cx, cy), glow_r + 4, glow_color, 1)

    # ── Nirodha: extra sacred geometry star ──
    if state_name == "Nirodha":
        for i in range(6):
            angle1 = rotation + (2 * math.pi * i / 6)
            angle2 = rotation + (2 * math.pi * (i + 2) / 6)
            x1 = int(cx + math.cos(angle1) * radius * 0.6)
            y1 = int(cy + math.sin(angle1) * radius * 0.6)
            x2 = int(cx + math.cos(angle2) * radius * 0.6)
            y2 = int(cy + math.sin(angle2) * radius * 0.6)
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # ── Blend overlay onto frame ──
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)