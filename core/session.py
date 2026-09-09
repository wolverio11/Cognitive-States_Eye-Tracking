# core/session.py

import time
import json
from collections import deque
from datetime import datetime


# ─── Session Class ────────────────────────────────────────────────────────────
class Session:
    """
    Tracks a full meditation session:
      - Duration
      - Score history (rolling + full)
      - Peak state reached
      - Average score
      - State time distribution (how long spent in each state)
    """

    ROLLING_WINDOW = 300   # seconds — for rolling average (last 5 min)

    def __init__(self, name: str = None):
        self.name          = name or datetime.now().strftime("Session_%Y%m%d_%H%M%S")
        self.start_time    = time.time()
        self.end_time      = None
        self.is_active     = True

        # Score tracking
        self._history      = []          # list of (timestamp, score, state_name)
        self._rolling      = deque()     # (timestamp, score) for rolling window

        # Summary stats
        self.peak_score    = 0.0
        self.peak_state    = "Vikshipta"
        self.average_score = 0.0

        # Time per state (seconds)
        self.state_time    = {
            "Vikshipta": 0.0,
            "Kshipta":   0.0,
            "Mudha":     0.0,
            "Ekagra":    0.0,
            "Nirodha":   0.0,
        }
        self._last_tick    = time.time()
        self._last_state   = "Vikshipta"

    # ── public API ────────────────────────────────────────────────────────────
    def tick(self, score: float, state_name: str):
        """Call every frame with the latest score and state."""
        if not self.is_active:
            return

        now = time.time()

        # Accumulate time in current state
        delta = now - self._last_tick
        self.state_time[self._last_state] = \
            self.state_time.get(self._last_state, 0.0) + delta
        self._last_tick  = now
        self._last_state = state_name

        # Record history
        self._history.append((now, score, state_name))

        # Rolling window
        self._rolling.append((now, score))
        cutoff = now - self.ROLLING_WINDOW
        while self._rolling and self._rolling[0][0] < cutoff:
            self._rolling.popleft()

        # Update peak
        if score > self.peak_score:
            self.peak_score = score
            self.peak_state = state_name

        # Update average
        all_scores = [s for _, s, _ in self._history]
        self.average_score = sum(all_scores) / len(all_scores)

    def stop(self):
        """End the session."""
        self.end_time  = time.time()
        self.is_active = False
        # Flush last state time
        delta = self.end_time - self._last_tick
        self.state_time[self._last_state] = \
            self.state_time.get(self._last_state, 0.0) + delta

    @property
    def duration(self) -> float:
        """Session duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def duration_str(self) -> str:
        secs  = int(self.duration)
        mins  = secs // 60
        secs  = secs % 60
        return f"{mins:02d}:{secs:02d}"

    @property
    def rolling_average(self) -> float:
        """Average score over the last 5 minutes."""
        if not self._rolling:
            return 0.0
        return sum(s for _, s in self._rolling) / len(self._rolling)

    def summary(self) -> dict:
        total = self.duration
        return {
            "name":            self.name,
            "duration":        self.duration_str,
            "average_score":   round(self.average_score, 4),
            "peak_score":      round(self.peak_score, 4),
            "peak_state":      self.peak_state,
            "rolling_avg":     round(self.rolling_average, 4),
            "state_time":      {k: round(v, 1) for k, v in self.state_time.items()},
            "state_percent":   {
                k: round((v / total) * 100, 1) if total > 0 else 0.0
                for k, v in self.state_time.items()
            },
            "total_frames":    len(self._history),
        }

    def save(self, path: str = None):
        """Save session summary to a JSON file."""
        path = path or f"sessions/{self.name}.json"
        import os
        os.makedirs("sessions", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        print(f"Session saved → {path}")


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    session = Session(name="TestSession")
    print(f"Started: {session.name}")

    # Simulate 10 seconds of ticks
    states = ["Vikshipta", "Kshipta", "Mudha", "Ekagra", "Nirodha"]
    for i in range(50):
        fake_score = random.uniform(0, 1)
        fake_state = states[int(fake_score * 5) if fake_score < 1.0 else 4]
        session.tick(fake_score, fake_state)
        time.sleep(0.1)

    session.stop()

    summary = session.summary()
    print("\n── Session Summary ──────────────────────")
    for k, v in summary.items():
        print(f"  {k:<18}: {v}")

    session.save()