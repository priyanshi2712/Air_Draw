"""
main.py
-------
Air Drawing App
Draw on screen using nothing but your index finger, tracked live
through your webcam with MediaPipe + OpenCV.

GESTURES
    ☝️  Only INDEX finger up        -> Draw
    ✌️  INDEX + MIDDLE fingers up   -> Hover / select a color from the toolbar
    ✊  Fist (no fingers up)         -> Pen up (move without drawing)
    🖐️  All 5 fingers up            -> Clear the canvas

KEYBOARD
    s  -> save your drawing as a PNG
    c  -> clear the canvas
    q  -> quit
"""

import cv2
import numpy as np
import time
import os
from hand_tracker import HandTracker

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
CAM_WIDTH, CAM_HEIGHT = 1280, 720
BRUSH_THICKNESS = 8
ERASER_THICKNESS = 40
TOOLBAR_HEIGHT = 90

COLORS = [
    ("Red", (0, 0, 255)),
    ("Green", (0, 200, 0)),
    ("Blue", (255, 0, 0)),
    ("Yellow", (0, 220, 220)),
    ("Purple", (200, 0, 160)),
    ("Eraser", (0, 0, 0)),  # special-cased: erases instead of drawing
]

SAVE_DIR = os.path.join(os.path.expanduser("~"), "air_draw_saves")


def build_toolbar(width):
    """Pre-render the color-selection toolbar strip shown at the top."""
    toolbar = np.zeros((TOOLBAR_HEIGHT, width, 3), dtype=np.uint8)
    toolbar[:] = (40, 40, 40)

    swatch_w = width // len(COLORS)
    boxes = []  # (x1, y1, x2, y2, color_name, bgr)
    for i, (name, bgr) in enumerate(COLORS):
        x1 = i * swatch_w
        x2 = x1 + swatch_w
        color_to_draw = bgr if name != "Eraser" else (60, 60, 60)
        cv2.rectangle(toolbar, (x1 + 10, 15), (x2 - 10, TOOLBAR_HEIGHT - 15), color_to_draw, -1)
        cv2.putText(toolbar, name, (x1 + 12, TOOLBAR_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        boxes.append((x1, 0, x2, TOOLBAR_HEIGHT, name, bgr))
    return toolbar, boxes


def color_at(x, y, boxes):
    for x1, y1, x2, y2, name, bgr in boxes:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name, bgr
    return None, None


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check that no other app is using it "
              "and that your OS has granted camera permission to your terminal/IDE.")
        return

    tracker = HandTracker(max_hands=1)
    toolbar, boxes = build_toolbar(CAM_WIDTH)

    canvas = None  # created on first frame once we know the real frame size
    draw_color = COLORS[0][1]
    draw_color_name = COLORS[0][0]
    prev_point = None
    last_save_msg_time = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror, so it feels natural
        frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

        if canvas is None:
            canvas = np.zeros_like(frame)

        frame = tracker.find_hands(frame, draw=True)
        landmarks = tracker.get_landmark_positions(frame)

        status_text = "No hand detected"

        if landmarks:
            fingers = tracker.fingers_up(landmarks)
            index_tip = landmarks[8][1], landmarks[8][2]
            middle_tip = landmarks[12][1], landmarks[12][2]

            up_count = sum(fingers.values())
            index_up = fingers.get("index", False)
            middle_up = fingers.get("middle", False)
            only_index = index_up and not middle_up and up_count == 1
            two_finger_select = index_up and middle_up and up_count == 2
            fist = up_count == 0
            all_up = up_count == 5

            if all_up:
                canvas = np.zeros_like(frame)
                status_text = "Cleared canvas (open palm)"
                prev_point = None

            elif two_finger_select:
                status_text = "Selecting..."
                prev_point = None
                cv2.circle(frame, index_tip, 12, (255, 255, 255), 2)
                if index_tip[1] < TOOLBAR_HEIGHT:
                    name, bgr = color_at(index_tip[0], index_tip[1], boxes)
                    if name:
                        draw_color_name = name
                        draw_color = bgr

            elif only_index and index_tip[1] > TOOLBAR_HEIGHT:
                status_text = f"Drawing ({draw_color_name})"
                if prev_point is None:
                    prev_point = index_tip
                thickness = ERASER_THICKNESS if draw_color_name == "Eraser" else BRUSH_THICKNESS
                cv2.line(canvas, prev_point, index_tip, draw_color, thickness)
                prev_point = index_tip
                cv2.circle(frame, index_tip, thickness // 2, draw_color, -1)

            elif fist:
                status_text = "Pen up"
                prev_point = None

            else:
                status_text = "Hand detected"
                prev_point = None

        else:
            prev_point = None

        # Merge canvas onto the live frame: turn drawn pixels into a mask,
        # black-out that region on the frame, then add the canvas colors in.
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        combined = cv2.add(frame_bg, canvas_fg)

        # Toolbar + status bar
        combined[0:TOOLBAR_HEIGHT, 0:CAM_WIDTH] = toolbar
        cv2.putText(combined, status_text, (10, CAM_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, "s: save   c: clear   q: quit", (10, CAM_HEIGHT - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        if time.time() - last_save_msg_time < 1.5:
            cv2.putText(combined, "Saved!", (CAM_WIDTH - 160, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Air Draw - press q to quit", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros_like(frame)
        elif key == ord('s'):
            filename = os.path.join(SAVE_DIR, f"drawing_{int(time.time())}.png")
            cv2.imwrite(filename, canvas)
            print(f"Saved drawing to {filename}")
            last_save_msg_time = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# python main.py
