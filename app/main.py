# app/main.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
from core.face_mesh   import FaceMeshTracker
from core.gaze_tracker import GazeTracker
from core.scorer      import Scorer
from core.session     import Session

from components.hud           import draw_hud
from components.mandala       import draw_mandala
from components.score_bar     import draw_score_bar
from components.session_chart import draw_session_panel, draw_end_screen
from components.gaze_overlay  import draw_gaze_overlay

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH    = "models/face_landmarker.task"
WINDOW_FACE   = "IKO - Face Tracker"
WINDOW_SCENE  = "IKO - Gaze Scene"
WARMUP_FRAMES = 60

# ─── Camera Sources ───────────────────────────────────────────────────────────
# Laptop webcam — tracks your face
FACE_CAM = 0

# DroidCam on forehead — sees the scene
# Option A: virtual webcam
SCENE_CAM = 1
# Option B: IP stream
# SCENE_CAM = "http://192.168.x.x:4747/video"


# ─── Main Loop ────────────────────────────────────────────────────────────────
def main():
    tracker = FaceMeshTracker(model_path=MODEL_PATH)
    gazer   = GazeTracker()
    scorer  = Scorer()
    session = Session()

    # ── Open both cameras ──
    print(f"Opening face cam  : {FACE_CAM}")
    print(f"Opening scene cam : {SCENE_CAM}")

    face_cap  = cv2.VideoCapture(FACE_CAM)
    scene_cap = cv2.VideoCapture(SCENE_CAM)

    if not face_cap.isOpened():
        print(f"ERROR: Could not open face camera: {FACE_CAM}")
        return
    if not scene_cap.isOpened():
        print(f"ERROR: Could not open scene camera: {SCENE_CAM}")
        print("Running without scene camera...")
        scene_cap = None

    print(f"Session started: {session.name}")
    print("Press  S  to stop   |   Q  to quit")

    frame_count   = 0
    session_ended = False
    end_frame     = None

    while True:
        # ── Read face frame ──
        ret_face, face_frame = face_cap.read()
        if not ret_face:
            print("WARNING: Lost face camera feed...")
            time.sleep(0.1)
            continue

        # ── Read scene frame ──
        scene_frame = None
        if scene_cap is not None:
            ret_scene, scene_frame = scene_cap.read()
            if not ret_scene:
                scene_frame = None

        if not session_ended:
            frame_count += 1
            warmup_done  = frame_count > WARMUP_FRAMES

            # ── Core pipeline ──
            face_frame = tracker.process(face_frame)

            # ── Gaze tracking ──
            if tracker.last_landmarks is not None:
                h, w = face_frame.shape[:2]
                gazer.update(tracker.last_landmarks, w, h)

            if warmup_done:
                scorer.update(tracker.blink_rate, tracker.gaze_variance)
                session.tick(scorer.score, scorer.state["name"])

            # ── Draw face window components ──
            draw_hud(face_frame, tracker, scorer, warmup_done)

            if warmup_done:
                draw_mandala(face_frame, scorer.score, scorer.state["name"])
                draw_score_bar(face_frame, scorer.score, scorer.state["name"])
                draw_session_panel(face_frame, session)

            # ── Draw scene window with gaze dot ──
            if scene_frame is not None and warmup_done:
                draw_gaze_overlay(scene_frame, gazer.pitch, gazer.yaw)

        else:
            face_frame = draw_end_screen(end_frame.copy(), session)

        # ── Show windows ──
        cv2.imshow(WINDOW_FACE, face_frame)
        if scene_frame is not None:
            cv2.imshow(WINDOW_SCENE, scene_frame)

        # ── Key handling ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and not session_ended:
            session.stop()
            session.save()
            end_frame     = face_frame.copy()
            session_ended = True
            print("\n── Session Summary ──────────────────")
            for k, v in session.summary().items():
                print(f"  {k:<18}: {v}")

    face_cap.release()
    if scene_cap is not None:
        scene_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()