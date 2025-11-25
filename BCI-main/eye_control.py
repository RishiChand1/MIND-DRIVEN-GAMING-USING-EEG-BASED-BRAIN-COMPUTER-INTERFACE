import cv2
import pyautogui
import numpy as np
from collections import deque
import time

# --- Parameters (tweak if needed) ---
CAM_INDEX = 0
SMOOTH_BUFFER = 6            # number of points for moving average smoothing
CLICK_COOLDOWN = 0.8         # seconds between clicks
AUTO_CALIBRATE_FRAMES = 50   # frames to estimate open-eye baseline
DARK_RATIO_FACTOR = 0.55     # threshold = baseline * DARK_RATIO_FACTOR

# --- Init ---
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

smooth_queue = deque(maxlen=SMOOTH_BUFFER)
last_click_time = 0.0

print("Starting. Please sit comfortably facing the camera.")
print("Auto-calibrating open-eyes baseline (keep eyes open) for ~2 seconds...")

# --- Auto-calibrate open-eye dark ratio baseline ---
open_ratios = []
count = 0
while count < AUTO_CALIBRATE_FRAMES:
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        cv2.imshow("Calibrating... (keep eyes open)", frame)
        cv2.waitKey(1)
        continue
    x, y, w, h = faces[0]  # use first face
    # approximate eye region relative to face
    ex = int(w * 0.13); ey = int(h * 0.20); ew = int(w * 0.36); eh = int(h * 0.20)
    left_eye = gray[y + ey : y + ey + eh, x + ex : x + ex + ew]
    right_eye = gray[y + ey : y + ey + eh, x + int(w*0.5) : x + int(w*0.5) + ew]

    def dark_ratio(eye_roi):
        if eye_roi.size == 0: 
            return None
        blur = cv2.GaussianBlur(eye_roi, (7,7), 0)
        # adaptive threshold is more robust to lighting
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
        cnt = cv2.countNonZero(th)
        return cnt / (eye_roi.shape[0] * eye_roi.shape[1])

    lr = dark_ratio(left_eye)
    rr = dark_ratio(right_eye)
    if lr is not None and rr is not None:
        open_ratios.append((lr + rr) / 2.0)
        count += 1
    cv2.imshow("Calibrating... (keep eyes open)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if len(open_ratios) == 0:
    baseline = 0.12  # fallback conservative value
else:
    baseline = np.mean(open_ratios)
blink_threshold = baseline * DARK_RATIO_FACTOR
print(f"Calibration complete. baseline={baseline:.3f}, blink_threshold={blink_threshold:.3f}")
cv2.destroyAllWindows()

# --- Main loop ---
print("Eye-gaze control started. Look to move cursor; blink wwto click. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    gaze_point = None
    blink_detected = False
    debug_text = ""

    if len(faces) > 0:
        x, y, w, h = faces[0]

        # define left and right eye ROIs by proportions of face
        ex = int(w * 0.13); ey = int(h * 0.20); ew = int(w * 0.36); eh = int(h * 0.20)
        lx, ly = x + ex, y + ey
        rx, ry = x + int(w*0.5), y + ey

        left_eye = gray[ly:ly+eh, lx:lx+ew]
        right_eye = gray[ry:ry+eh, rx:rx+ew]

        # function to find pupil centroid in eye roi
        def pupil_centroid(eye_roi):
            if eye_roi.size == 0:
                return None, None
            # equalize + blur
            eq = cv2.equalizeHist(eye_roi)
            blur = cv2.GaussianBlur(eq, (7,7), 0)
            # binary inverse: pupil/iris become white
            _, th = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # find largest contour
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, None
            cnt = max(contours, key=cv2.contourArea)
            ((cx, cy), r) = cv2.minEnclosingCircle(cnt)
            # reject tiny contours
            if r < 2:
                return None, None
            return int(cx), int(cy), th

        # compute pupil centroids
        lc = pupil_centroid(left_eye)
        rc = pupil_centroid(right_eye)

        # compute dark ratio (for blink detection)
        def dark_ratio(eye_roi):
            if eye_roi.size == 0:
                return None
            blur = cv2.GaussianBlur(eye_roi, (7,7), 0)
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
            return cv2.countNonZero(th) / (eye_roi.shape[0] * eye_roi.shape[1]), th

        lr, lth = dark_ratio(left_eye)
        rr, rth = dark_ratio(right_eye)

        # combine pupil centroids (prefer average if both present)
        pupil_positions = []
        if lc[0] is not None:
            pupil_positions.append((lc[0]+lx, lc[1]+ly))
            cv2.circle(frame, (lx+lc[0], ly+lc[1]), 3, (0,255,0), -1)
        if rc[0] is not None:
            pupil_positions.append((rx+rc[0], ry+rc[1]))
            cv2.circle(frame, (rx+rc[0], ry+rc[1]), 3, (0,255,0), -1)

        if pupil_positions:
            avg_px = int(np.mean([p[0] for p in pupil_positions]))
            avg_py = int(np.mean([p[1] for p in pupil_positions]))
            # map camera coords to screen
            gaze_x = np.interp(avg_px, [0, frame.shape[1]], [0, screen_w])
            gaze_y = np.interp(avg_py, [0, frame.shape[0]], [0, screen_h])
            smooth_queue.append((gaze_x, gaze_y))
            avg_x = int(np.mean([p[0] for p in smooth_queue]))
            avg_y = int(np.mean([p[1] for p in smooth_queue]))
            pyautogui.moveTo(avg_x, avg_y, _pause=False)
            gaze_point = (avg_x, avg_y)

        # blink detection: use min of left/right dark ratio (both eyes must be low)
        if lr is not None and rr is not None:
            avg_ratio = (lr + rr) / 2.0
            debug_text = f"ratio={avg_ratio:.3f}"
            if avg_ratio < blink_threshold:
                # blink candidate: check cooldown and confirm that it persists for a couple frames
                current_time = time.time()
                if current_time - last_click_time > CLICK_COOLDOWN:
                    # we may require the low ratio across consecutive frames: simple approach: click immediately but set cooldown
                    pyautogui.click()
                    last_click_time = current_time
                    blink_detected = True

            # draw debug thumbnails of eye threshold (optional)
            # show the binary thumbnail for observation
            if lth is not None:
                small = cv2.resize(lth, (80,50))
                frame[5:5+50, 5:5+80] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            if rth is not None:
                small2 = cv2.resize(rth, (80,50))
                frame[5:5+50, 90:90+80] = cv2.cvtColor(small2, cv2.COLOR_GRAY2BGR)

        # draw rectangles for eyes
        cv2.rectangle(frame, (lx, ly), (lx+ew, ly+eh), (255,0,0), 1)
        cv2.rectangle(frame, (rx, ry), (rx+ew, ry+eh), (255,0,0), 1)

    # overlays
    if gaze_point:
        cv2.circle(frame, (int(np.interp(gaze_point[0],[0,screen_w],[0,frame.shape[1]])),
                           int(np.interp(gaze_point[1],[0,screen_h],[0,frame.shape[0]]))), 6, (0,255,255), 2)
    cv2.putText(frame, f"BlinkThresh:{blink_threshold:.3f}", (10, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    if debug_text:
        cv2.putText(frame, debug_text, (10, frame.shape[0]-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    cv2.imshow("Eye Gaze Click (Improved)", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('+'):
        DARK_RATIO_FACTOR *= 1.05
        blink_threshold = baseline * DARK_RATIO_FACTOR
    elif key == ord('-'):
        DARK_RATIO_FACTOR *= 0.95
        blink_threshold = baseline * DARK_RATIO_FACTOR

cap.release()
cv2.destroyAllWindows()
