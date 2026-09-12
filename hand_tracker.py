"""
hand_tracker.py
----------------
Hand tracking using MediaPipe's newer Tasks API (HandLandmarker).

Why not `mp.solutions.hands`? That's the old legacy API, and recent
MediaPipe builds (especially on newer Python versions like 3.13/3.14)
have dropped it, causing:
    AttributeError: module 'mediapipe' has no attribute 'solutions'
The Tasks API below is the actively maintained replacement and works
across current MediaPipe versions.

On first run this will auto-download the small hand-landmark model
file (hand_landmarker.task, ~8MB) from Google's model storage into
the project folder, then reuse it on later runs.
"""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# The 21-point hand skeleton connections, used for manual drawing since
# the Tasks API (unlike the old solutions API) doesn't ship a drawing helper.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand-tracking model (first run only, ~8MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.6, tracking_confidence=0.6):
        _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.results = None
        self._frame_count = 0

    def find_hands(self, frame, draw=True):
        """Run detection on a BGR frame and optionally draw the skeleton."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires an increasing timestamp per frame (in ms)
        timestamp_ms = self._frame_count * 33  # ~30fps assumption
        self._frame_count += 1

        self.results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if self.results.hand_landmarks and draw:
            h, w = frame.shape[:2]
            for hand_landmarks in self.results.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
                for x, y in pts:
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        return frame

    def get_landmark_positions(self, frame, hand_index=0):
        """Return a list of (id, x, y) pixel coordinates for one detected hand."""
        positions = []
        if self.results and self.results.hand_landmarks:
            if hand_index >= len(self.results.hand_landmarks):
                return positions
            hand = self.results.hand_landmarks[hand_index]
            h, w = frame.shape[:2]
            for idx, lm in enumerate(hand):
                positions.append((idx, int(lm.x * w), int(lm.y * h)))
        return positions

    def fingers_up(self, landmarks):
        """
        Given the 21 landmark positions, return a dict telling us which
        fingers are extended, e.g. {"thumb": False, "index": True, ...}.
        """
        if not landmarks:
            return {}

        lm = {idx: (x, y) for idx, x, y in landmarks}
        fingers = {}

        fingers["thumb"] = lm[4][0] > lm[3][0] if lm[17][0] < lm[5][0] else lm[4][0] < lm[3][0]

        for name, tip_id in [("index", 8), ("middle", 12), ("ring", 16), ("pinky", 20)]:
            pip_id = tip_id - 2
            fingers[name] = lm[tip_id][1] < lm[pip_id][1]

        return fingers
