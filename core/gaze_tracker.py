# core/gaze_tracker.py

import cv2
import numpy as np
from collections import deque

# ─── Landmark indices ─────────────────────────────────────────────────────────
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Head pose reference points (stable landmarks)
NOSE_TIP    = 1
CHIN        = 152
LEFT_CORNER = 263
RIGHT_CORNER = 33
LEFT_MOUTH  = 287
RIGHT_MOUTH = 57

SMOOTH_WINDOW = 8


class GazeTracker:

    def __init__(self):
        self.pitch          = 0.0
        self.yaw            = 0.0
        self._pitch_history = deque(maxlen=SMOOTH_WINDOW)
        self._yaw_history   = deque(maxlen=SMOOTH_WINDOW)
        self._baseline_set  = False
        self._baseline_yaw  = 0.0
        self._baseline_pitch = 0.0
        self._calibration_frames = deque(maxlen=30)  # 1 sec of frames

    def update(self, landmarks, img_w: int, img_h: int):
        if landmarks is None:
            return

        # ── Get iris centers ──
        left_iris  = self._mean_point(landmarks, LEFT_IRIS,  img_w, img_h)
        right_iris = self._mean_point(landmarks, RIGHT_IRIS, img_w, img_h)

        # ── Get eye centers (midpoint of eye corners) ──
        left_eye_center  = self._mean_point(landmarks, LEFT_EYE,  img_w, img_h)
        right_eye_center = self._mean_point(landmarks, RIGHT_EYE, img_w, img_h)

        # ── Get head reference points ──
        nose  = self._point(landmarks, NOSE_TIP,    img_w, img_h)
        chin  = self._point(landmarks, CHIN,         img_w, img_h)
        l_cor = self._point(landmarks, LEFT_CORNER,  img_w, img_h)
        r_cor = self._point(landmarks, RIGHT_CORNER, img_w, img_h)

        # ── Face size for normalization ──
        face_w = max(np.linalg.norm(l_cor - r_cor), 1)
        face_h = max(np.linalg.norm(nose  - chin),  1)

        # ── Iris offset from eye center, normalized by face size ──
        left_offset  = left_iris  - left_eye_center
        right_offset = right_iris - right_eye_center

        raw_yaw   = ((left_offset[0] + right_offset[0]) / 2.0) / face_w
        raw_pitch = ((left_offset[1] + right_offset[1]) / 2.0) / face_h

        # ── Auto calibration (first 30 frames = forward gaze baseline) ──
        if not self._baseline_set:
            self._calibration_frames.append((raw_yaw, raw_pitch))
            if len(self._calibration_frames) == 30:
                self._baseline_yaw   = np.mean([f[0] for f in self._calibration_frames])
                self._baseline_pitch = np.mean([f[1] for f in self._calibration_frames])
                self._baseline_set   = True
            return

        # ── Subtract baseline (center gaze at 0,0) ──
        raw_yaw   -= self._baseline_yaw
        raw_pitch -= self._baseline_pitch

        # ── Scale to -1.0 to 1.0 ──
        raw_yaw   = np.clip(raw_yaw   * 15, -1.0, 1.0)
        raw_pitch = np.clip(raw_pitch * 15, -1.0, 1.0)

        # ── Smooth ──
        self._yaw_history.append(raw_yaw)
        self._pitch_history.append(raw_pitch)

        self.yaw   = float(np.mean(self._yaw_history))
        self.pitch = float(np.mean(self._pitch_history))

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _mean_point(self, landmarks, indices, img_w, img_h):
        pts = np.array([
            [landmarks[i].x * img_w, landmarks[i].y * img_h]
            for i in indices
        ])
        return pts.mean(axis=0)

    def _point(self, landmarks, index, img_w, img_h):
        return np.array([
            landmarks[index].x * img_w,
            landmarks[index].y * img_h
        ])