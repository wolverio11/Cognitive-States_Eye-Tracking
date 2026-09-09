# core/scorer.py

import numpy as np

# ─── Chitta State Definitions ─────────────────────────────────────────────────
CHITTA_STATES = [
    {"name": "Vikshipta", "sanskrit": "विक्षिप्त", "range": (0.00, 0.25), "desc": "Scattered mind"},
    {"name": "Kshipta",   "sanskrit": "क्षिप्त",   "range": (0.25, 0.45), "desc": "Agitated mind"},
    {"name": "Mudha",     "sanskrit": "मूढ",        "range": (0.45, 0.60), "desc": "Dull mind"},
    {"name": "Ekagra",    "sanskrit": "एकाग्र",     "range": (0.60, 0.80), "desc": "One-pointed focus"},
    {"name": "Nirodha",   "sanskrit": "निरोध",      "range": (0.80, 1.00), "desc": "Witness state"},
]

# ─── Scoring Weights ──────────────────────────────────────────────────────────
BLINK_WEIGHT = 0.4
GAZE_WEIGHT  = 0.6

# ─── Calibration Ranges (tweak after testing) ─────────────────────────────────
# Blink rate:     ~20 blinks/min = normal, ~5 = very still, ~40 = very restless
BLINK_MIN = 4.0    # blinks/min → score 1.0 (most still)
BLINK_MAX = 35.0   # blinks/min → score 0.0 (most scattered)

# Gaze variance:  ~0 = perfectly still, ~200+ = very restless
GAZE_MIN  = 0.0    # px² → score 1.0
GAZE_MAX  = 200.0  # px² → score 0.0


# ─── Scorer Class ─────────────────────────────────────────────────────────────
class Scorer:
    """
    Converts raw blink_rate + gaze_variance into:
      - score      : float 0.0 – 1.0
      - state      : dict  (name, sanskrit, desc)
      - blink_score: float component
      - gaze_score : float component
    """

    def __init__(self):
        self.score       = 0.0
        self.state       = CHITTA_STATES[0]
        self.blink_score = 0.0
        self.gaze_score  = 0.0

    def update(self, blink_rate: float, gaze_variance: float):
        self.blink_score = self._normalize_blink(blink_rate)
        self.gaze_score  = self._normalize_gaze(gaze_variance)

        self.score = (
            BLINK_WEIGHT * self.blink_score +
            GAZE_WEIGHT  * self.gaze_score
        )
        self.score = float(np.clip(self.score, 0.0, 1.0))
        self.state = self._get_state(self.score)

    def _normalize_blink(self, blink_rate: float) -> float:
        """Lower blink rate → higher score (more still)."""
        if blink_rate <= BLINK_MIN:
            return 1.0
        if blink_rate >= BLINK_MAX:
            return 0.0
        return 1.0 - (blink_rate - BLINK_MIN) / (BLINK_MAX - BLINK_MIN)

    def _normalize_gaze(self, gaze_variance: float) -> float:
        """Lower variance → higher score (more still)."""
        if gaze_variance <= GAZE_MIN:
            return 1.0
        if gaze_variance >= GAZE_MAX:
            return 0.0
        return 1.0 - (gaze_variance - GAZE_MIN) / (GAZE_MAX - GAZE_MIN)

    def _get_state(self, score: float) -> dict:
        for state in CHITTA_STATES:
            low, high = state["range"]
            if low <= score < high:
                return state
        return CHITTA_STATES[-1]  # Nirodha if score == 1.0

    def summary(self) -> dict:
        return {
            "score":       round(self.score, 4),
            "state":       self.state["name"],
            "sanskrit":    self.state["sanskrit"],
            "desc":        self.state["desc"],
            "blink_score": round(self.blink_score, 4),
            "gaze_score":  round(self.gaze_score, 4),
        }


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scorer = Scorer()

    test_cases = [
        (30, 180),   # very scattered → Vikshipta
        (20, 100),   # average → Kshipta
        (12,  80),   # slightly better → Mudha
        ( 7,  30),   # focused → Ekagra
        ( 4,   5),   # near stillness → Nirodha
    ]

    print(f"{'Blink/min':<12} {'Gaze Var':<12} {'Score':<8} {'State':<12} {'Desc'}")
    print("─" * 60)
    for blink, gaze in test_cases:
        scorer.update(blink, gaze)
        s = scorer.summary()
        print(f"{blink:<12} {gaze:<12} {s['score']:<8} {s['state']:<12} {s['desc']}")