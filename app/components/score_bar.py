# app/components/score_bar.py

import cv2

COLOR_WHITE = (255, 255, 255)

STATE_COLORS = {
    "Vikshipta": (0, 0, 255),
    "Kshipta":   (0, 140, 255),
    "Mudha":     (0, 255, 255),
    "Ekagra":    (0, 255, 120),
    "Nirodha":   (255, 255, 255),
}


def draw_score_bar(frame, score: float, state_name: str) -> None:
    h, w  = frame.shape[:2]
    color = STATE_COLORS.get(state_name, COLOR_WHITE)

    bar_x, bar_y = 10, h - 40
    bar_w        = w - 20
    bar_h        = 18

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)

    # Fill
    fill_w = int(bar_w * max(0.0, min(score, 1.0)))
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

    # Border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_WHITE, 1)

    # Score label
    label = f"Score: {score:.2f}"
    cv2.putText(frame, label, (bar_x + 4, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(frame, label, (bar_x + 4, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

    # State label right side
    tw = cv2.getTextSize(state_name.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cv2.putText(frame, state_name.upper(),
                (bar_x + bar_w - tw - 6, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(frame, state_name.upper(),
                (bar_x + bar_w - tw - 6, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)