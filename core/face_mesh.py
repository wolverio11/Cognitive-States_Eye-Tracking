# core/face_mesh.py

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ─── Landmark Indices ─────────────────────────────────────────────────────────
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    pts = np.array([
        [landmarks[i].x * img_w, landmarks[i].y * img_h]
        for i in eye_indices
    ])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)


def iris_center(landmarks, iris_indices, img_w, img_h):
    pts = np.array([
        [landmarks[i].x * img_w, landmarks[i].y * img_h]
        for i in iris_indices
    ])
    return pts.mean(axis=0)


def draw_landmarks_on_frame(frame, face_landmarks_list):
    """Draw face mesh using cv2 directly."""
    h, w = frame.shape[:2]
    for face_landmarks in face_landmarks_list:
        for lm in face_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


# ─── FaceMeshTracker ──────────────────────────────────────────────────────────
class FaceMeshTracker:

    EAR_THRESHOLD     = 0.21
    EAR_CONSEC_FRAMES = 2
    HISTORY_SECONDS   = 60
    GAZE_HISTORY      = 90

    def __init__(self, model_path="models/face_landmarker.task"):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self.landmarker     = vision.FaceLandmarker.create_from_options(options)

        self._ear_consec    = 0
        self._blink_times   = deque()
        self._gaze_history  = deque(maxlen=self.GAZE_HISTORY)
        self.blink_rate     = 0.0
        self.gaze_variance  = 0.0
        self.ear            = 0.0
        self.last_landmarks = None        # ← used by gaze_tracker.py

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result    = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            self.last_landmarks = None
            return frame

        lm = result.face_landmarks[0]
        self.last_landmarks = lm             # ← save for gaze_tracker

        # ── EAR & blink detection ──
        left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
        right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        self.ear  = (left_ear + right_ear) / 2.0

        if self.ear < self.EAR_THRESHOLD:
            self._ear_consec += 1
        else:
            if self._ear_consec >= self.EAR_CONSEC_FRAMES:
                self._blink_times.append(time.time())
            self._ear_consec = 0

        self._update_blink_rate()

        # ── Iris gaze tracking ──
        left_c  = iris_center(lm, LEFT_IRIS,  w, h)
        right_c = iris_center(lm, RIGHT_IRIS, w, h)
        self._gaze_history.append((left_c + right_c) / 2.0)
        self._update_gaze_variance()

        # ── Draw landmarks ──
        draw_landmarks_on_frame(frame, result.face_landmarks)
        return frame

    def _update_blink_rate(self):
        now    = time.time()
        cutoff = now - self.HISTORY_SECONDS
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        elapsed = (now - self._blink_times[0]) if self._blink_times else self.HISTORY_SECONDS
        elapsed = min(elapsed, self.HISTORY_SECONDS)
        self.blink_rate = (len(self._blink_times) / elapsed * 60.0) if elapsed > 0 else 0.0

    def _update_gaze_variance(self):
        if len(self._gaze_history) < 2:
            self.gaze_variance = 0.0
            return
        pts = np.array(self._gaze_history)
        self.gaze_variance = float(np.var(pts, axis=0).sum())


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tracker = FaceMeshTracker()
    cap = cv2.VideoCapture(0)

    print("Running... Press Q to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = tracker.process(frame)

        eye_status = "CLOSED" if tracker.ear < FaceMeshTracker.EAR_THRESHOLD else "OPEN"
        eye_color  = (0, 0, 255) if eye_status == "CLOSED" else (0, 255, 0)

        cv2.putText(frame, f"Eyes: {eye_status}",               (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
        cv2.putText(frame, f"EAR: {tracker.ear:.3f}",           (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks/min: {tracker.blink_rate:.1f}", (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gaze Var: {tracker.gaze_variance:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("IKO - Face Mesh", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()