# app/components/hud.py

import cv2

# ─── Colors (BGR) ─────────────────────────────────────────────────────────────
COLOR_WHITE  = (255, 255, 255)
COLOR_RED    = (0, 0, 255)
COLOR_GREEN  = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)

STATE_COLORS = {
    "Vikshipta": (0, 0, 255),
    "Kshipta":   (0, 140, 255),
    "Mudha":     (0, 255, 255),
    "Ekagra":    (0, 255, 120),
    "Nirodha":   (255, 255, 255),
}


def draw_hud(frame, tracker, scorer, warmup_done: bool) -> None:
    """
    Draws top-left HUD:
      - Calibrating message during warmup
      - State name + description
      - Eye status (OPEN/CLOSED)
      - EAR, blink rate, gaze variance
    """
    if not warmup_done:
        cv2.putText(frame, "Calibrating...",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_YELLOW, 2)
        return

    state_name = scorer.state["name"]
    desc       = scorer.state["desc"]
    color      = STATE_COLORS.get(state_name, COLOR_WHITE)

    # ── State name ──
    cv2.putText(frame, state_name,
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # ── State description ──
    cv2.putText(frame, f"({desc})",
                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # ── Eye status ──
    eye_status = "CLOSED" if tracker.ear < 0.21 else "OPEN"
    eye_color  = COLOR_RED if eye_status == "CLOSED" else COLOR_GREEN
    cv2.putText(frame, f"Eyes: {eye_status}",
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)

    # ── Metrics ──
    cv2.putText(frame, f"EAR: {tracker.ear:.3f}",
                (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)
    cv2.putText(frame, f"Blinks/min: {tracker.blink_rate:.1f}",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)
    cv2.putText(frame, f"Gaze Var: {tracker.gaze_variance:.2f}",
                (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)