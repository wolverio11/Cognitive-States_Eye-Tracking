# app/components/session_chart.py

import cv2
import numpy as np

# ─── Colors (BGR) ─────────────────────────────────────────────────────────────
COLOR_WHITE  = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_BLUE   = (255, 180, 0)
COLOR_BLACK  = (0, 0, 0)
COLOR_GRAY   = (80, 80, 80)

STATE_COLORS = {
    "Vikshipta": (0, 0, 255),
    "Kshipta":   (0, 140, 255),
    "Mudha":     (0, 255, 255),
    "Ekagra":    (0, 255, 120),
    "Nirodha":   (255, 255, 255),
}


# ─── Session Panel (top right, shown during live session) ─────────────────────
def draw_session_panel(frame, session) -> None:
    """
    Draws top-right panel showing:
      - Session duration
      - Average score
      - Peak score + state
      - Mini score graph (last 60 readings)
    """
    h, w = frame.shape[:2]

    # ── Text info ──
    cv2.putText(frame, f"Duration: {session.duration_str}",
                (w - 210, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
    cv2.putText(frame, f"Avg: {session.average_score:.2f}",
                (w - 210, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 1)
    cv2.putText(frame, f"Peak: {session.peak_score:.2f} ({session.peak_state})",
                (w - 210, 89), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLUE, 1)

    # ── Mini score graph ──
    if len(session._history) < 2:
        return

    # Sample last 60 score readings
    history  = session._history[-60:]
    scores   = [s for _, s, _ in history]
    states   = [st for _, _, st in history]

    graph_x  = w - 210
    graph_y  = 105
    graph_w  = 195
    graph_h  = 45

    # Background
    cv2.rectangle(frame,
                  (graph_x, graph_y),
                  (graph_x + graph_w, graph_y + graph_h),
                  (30, 30, 30), -1)
    cv2.rectangle(frame,
                  (graph_x, graph_y),
                  (graph_x + graph_w, graph_y + graph_h),
                  COLOR_GRAY, 1)

    # Midline at 0.5
    mid_y = graph_y + graph_h // 2
    cv2.line(frame,
             (graph_x, mid_y),
             (graph_x + graph_w, mid_y),
             COLOR_GRAY, 1)

    # Plot score line
    n = len(scores)
    for i in range(1, n):
        x1 = graph_x + int((i - 1) / max(n - 1, 1) * graph_w)
        x2 = graph_x + int(i       / max(n - 1, 1) * graph_w)
        y1 = graph_y + graph_h - int(scores[i - 1] * graph_h)
        y2 = graph_y + graph_h - int(scores[i]     * graph_h)
        color = STATE_COLORS.get(states[i], COLOR_WHITE)
        cv2.line(frame, (x1, y1), (x2, y2), color, 1)

    # Graph label
    cv2.putText(frame, "score history",
                (graph_x + 2, graph_y + graph_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_GRAY, 1)


# ─── End Screen (shown after session stops) ───────────────────────────────────
def draw_end_screen(frame, session) -> None:
    """
    Full session summary overlay:
      - Duration, avg score, peak state
      - State time distribution bars
      - Score history graph
    """
    # Dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    summary = session.summary()
    h, w    = frame.shape[:2]
    cx      = w // 2

    # ── Title ──
    _centered_text(frame, "Session Complete", cx, 45, 0.95, COLOR_WHITE, 2)

    # ── Stats ──
    _centered_text(frame, f"Duration  :  {summary['duration']}",             cx, 90,  0.7, COLOR_WHITE,  1)
    _centered_text(frame, f"Avg Score :  {summary['average_score']}",        cx, 120, 0.7, COLOR_YELLOW, 1)
    _centered_text(frame, f"Peak      :  {summary['peak_score']}  ({summary['peak_state']})", cx, 150, 0.7, COLOR_BLUE, 1)

    # ── Divider ──
    cv2.line(frame, (40, 168), (w - 40, 168), COLOR_GRAY, 1)

    # ── State time bars ──
    _centered_text(frame, "Time in each state", cx, 190, 0.55, COLOR_GRAY, 1)

    states = ["Vikshipta", "Kshipta", "Mudha", "Ekagra", "Nirodha"]
    for i, s in enumerate(states):
        pct   = summary["state_percent"][s]
        secs  = summary["state_time"][s]
        color = STATE_COLORS[s]
        y     = 205 + i * 28

        bar_w = int((w - 40) * pct / 100)
        cv2.rectangle(frame, (20, y), (20 + max(bar_w, 4), y + 18), color, -1)

        # Label inside bar
        label = f"{s}  {pct}%  ({secs}s)"
        cv2.putText(frame, label, (25, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BLACK, 2)
        cv2.putText(frame, label, (25, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

    # ── Score history graph ──
    if len(session._history) > 1:
        _draw_score_graph(frame, session, x=20, y=355, gw=w - 40, gh=70)

    # ── Exit hint ──
    _centered_text(frame, "Press Q to exit", cx, h - 15, 0.5, COLOR_GRAY, 1)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _centered_text(frame, text, cx, y, scale, color, thickness):
    tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    cv2.putText(frame, text, (cx - tw // 2, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _draw_score_graph(frame, session, x, y, gw, gh):
    """Full-width score graph for end screen."""
    scores = [s for _, s, _ in session._history]
    states = [st for _, _, st in session._history]

    # Downsample to graph width
    n      = len(scores)
    step   = max(1, n // gw)
    scores = scores[::step]
    states = states[::step]
    n      = len(scores)

    # Background
    cv2.rectangle(frame, (x, y), (x + gw, y + gh), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + gw, y + gh), COLOR_GRAY, 1)

    # Gridlines at 0.25, 0.5, 0.75
    for lvl in [0.25, 0.5, 0.75]:
        gy = y + gh - int(lvl * gh)
        cv2.line(frame, (x, gy), (x + gw, gy), (50, 50, 50), 1)
        cv2.putText(frame, f"{lvl:.2f}", (x + 2, gy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_GRAY, 1)

    # Plot
    for i in range(1, n):
        x1 = x + int((i - 1) / max(n - 1, 1) * gw)
        x2 = x + int(i       / max(n - 1, 1) * gw)
        y1 = y + gh - int(scores[i - 1] * gh)
        y2 = y + gh - int(scores[i]     * gh)
        color = STATE_COLORS.get(states[i], COLOR_WHITE)
        cv2.line(frame, (x1, y1), (x2, y2), color, 1)

    # Label
    cv2.putText(frame, "Score History",
                (x + gw // 2 - 40, y + gh - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_GRAY, 1)