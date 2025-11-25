# hybrid_control.py
"""
Hybrid EEG + Eye Control System
--------------------------------
- EEG thread reads serial EEG signals, extracts features, loads trained model/scaler,
  predicts actions (e.g., move, jump), and sends them to a queue.
- Main thread runs the eye-tracking control using OpenCV.
- All keyboard/mouse actions (pyautogui) are done in the main thread.
"""

import cv2
import pyautogui
import numpy as np
from collections import deque
import time
import threading
import queue
import pickle
import serial
from scipy import signal
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# CONFIG
# -----------------------------
COM_PORT = 'COM5'         # Change to your Arduino's COM port
BAUD_RATE = 115200
SAMPLING_RATE = 512
EEG_BUFFER_LEN = 512

CAM_INDEX = 0
SMOOTH_BUFFER = 6
CLICK_COOLDOWN = 0.8
AUTO_CALIBRATE_FRAMES = 50
DARK_RATIO_FACTOR = 0.55  # Blink threshold scaling

# -----------------------------
# Shared Objects
# -----------------------------
action_queue = queue.Queue()   # Holds EEG-predicted actions for the main thread
stop_event = threading.Event()
pyautogui.FAILSAFE = False


# ======================================================
#                  EEG PROCESSING THREAD
# ======================================================

def setup_filters(sampling_rate):
    b_notch, a_notch = signal.iirnotch(50.0 / (0.5 * sampling_rate), 30.0)
    b_bandpass, a_bandpass = signal.butter(4, [0.5 / (0.5 * sampling_rate),
                                               30.0 / (0.5 * sampling_rate)], 'band')
    return b_notch, a_notch, b_bandpass, a_bandpass


def process_eeg_data(data, b_notch, a_notch, b_bandpass, a_bandpass):
    try:
        d = signal.filtfilt(b_notch, a_notch, data)
        d = signal.filtfilt(b_bandpass, a_bandpass, d)
        return d
    except Exception as e:
        print(f"[EEG] Filtering error: {e}")
        return data


def calculate_psd_features(segment, sampling_rate):
    f, psd_values = signal.welch(segment, fs=sampling_rate, nperseg=len(segment))
    bands = {'alpha': (8, 13), 'beta': (14, 30), 'theta': (4, 7), 'delta': (0.5, 3)}
    features = {}
    for band, (low, high) in bands.items():
        idx = np.where((f >= low) & (f <= high))
        features[f'E_{band}'] = np.sum(psd_values[idx])
    features['alpha_beta_ratio'] = (
        features['E_alpha'] / features['E_beta'] if features['E_beta'] > 0 else 0
    )
    return features


def calculate_additional_features(segment, sampling_rate):
    f, psd = signal.welch(segment, fs=sampling_rate, nperseg=len(segment))
    if np.sum(psd) == 0 or len(f) < 2:
        return {'peak_frequency': 0, 'spectral_centroid': 0, 'spectral_slope': 0}
    peak_frequency = f[np.argmax(psd)]
    spectral_centroid = np.sum(f * psd) / np.sum(psd)
    try:
        log_f = np.log(f[1:])
        log_psd = np.log(psd[1:])
        spectral_slope = np.polyfit(log_f, log_psd, 1)[0]
    except Exception:
        spectral_slope = 0
    return {'peak_frequency': peak_frequency,
            'spectral_centroid': spectral_centroid,
            'spectral_slope': spectral_slope}


def load_model_and_scaler(model_path='model.pkl', scaler_path='scaler.pkl'):
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return clf, scaler


def eeg_worker(action_queue, stop_event, com_port=COM_PORT, baud=BAUD_RATE):
    print("[EEG] Starting EEG worker thread...")

    try:
        ser = serial.Serial(com_port, baud, timeout=1)
        print(f"[EEG] Opened serial port {com_port} at {baud}")
    except Exception as e:
        print(f"[EEG] Could not open serial port {com_port}: {e}")
        return

    try:
        clf, scaler = load_model_and_scaler()
        print("[EEG] Model and scaler loaded.")
    except Exception as e:
        print(f"[EEG] Error loading model/scaler: {e}")
        ser.close()
        return

    b_notch, a_notch, b_bandpass, a_bandpass = setup_filters(SAMPLING_RATE)
    buffer = deque(maxlen=EEG_BUFFER_LEN)
    last_pred_time = 0.0
    min_prediction_interval = 0.5  # seconds between predictions

    while not stop_event.is_set():
        try:
            raw = ser.readline().decode('latin-1').strip()
            if not raw:
                continue

            try:
                val = float(raw)
            except:
                parts = [p for p in raw.split(',') if p.replace('.', '', 1).lstrip('-').isdigit()]
                if parts:
                    val = float(parts[0])
                else:
                    continue

            buffer.append(val)

            if len(buffer) == EEG_BUFFER_LEN and (time.time() - last_pred_time) > min_prediction_interval:
                arr = np.array(buffer)
                processed = process_eeg_data(arr, b_notch, a_notch, b_bandpass, a_bandpass)
                psd_f = calculate_psd_features(processed, SAMPLING_RATE)
                add_f = calculate_additional_features(processed, SAMPLING_RATE)
                features = {**psd_f, **add_f}
                df = pd.DataFrame([features])

                try:
                    X_scaled = scaler.transform(df)
                    prediction = clf.predict(X_scaled)[0]
                except Exception as e:
                    print(f"[EEG] Prediction error: {e}")
                    buffer.clear()
                    last_pred_time = time.time()
                    continue

                print(f"[EEG] Predicted class: {prediction}")

                # Map EEG classes to actions (edit this to suit your game)
                if prediction == 0:
                    action_queue.put(('press', 'space', 0.8))  # Jump
                elif prediction == 1:
                    action_queue.put(('press', 'w', 0.8))      # Move forward

                buffer.clear()
                last_pred_time = time.time()

        except Exception as e:
            print(f"[EEG] Loop error: {e}")
            continue

    print("[EEG] Stopping EEG worker...")
    try:
        ser.close()
    except:
        pass


# ======================================================
#                  EYE CONTROL LOOP
# ======================================================

def start_eye_control(action_queue, stop_event):
    print("Starting eye control. Please face the camera and keep eyes open for calibration.")
    screen_w, screen_h = pyautogui.size()
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                         "haarcascade_frontalface_default.xml")
    smooth_queue = deque(maxlen=SMOOTH_BUFFER)
    last_click_time = 0.0

    # Auto-calibrate open-eye ratio baseline
    open_ratios = []
    count = 0
    while count < AUTO_CALIBRATE_FRAMES and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            cv2.imshow("Calibrating... (keep eyes open)", frame)
            cv2.waitKey(1)
            continue
        x, y, w, h = faces[0]
        ex = int(w * 0.13); ey = int(h * 0.20)
        ew = int(w * 0.36); eh = int(h * 0.20)
        left_eye = gray[y + ey:y + ey + eh, x + ex:x + ex + ew]
        right_eye = gray[y + ey:y + ey + eh, x + int(w * 0.5):x + int(w * 0.5) + ew]

        def dark_ratio(eye_roi):
            if eye_roi.size == 0:
                return None
            blur = cv2.GaussianBlur(eye_roi, (7, 7), 0)
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
            stop_event.set()
            break

    baseline = np.mean(open_ratios) if len(open_ratios) > 0 else 0.12
    blink_threshold = baseline * DARK_RATIO_FACTOR
    cv2.destroyAllWindows()
    print(f"Calibration complete. baseline={baseline:.3f}, blink_threshold={blink_threshold:.3f}")
    print("Eye control started. Move your eyes to move cursor; blink to click. Press 'q' to quit.")

    active_keys = {}

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        gaze_point = None

        if len(faces) > 0:
            x, y, w, h = faces[0]
            ex = int(w * 0.13); ey = int(h * 0.20)
            ew = int(w * 0.36); eh = int(h * 0.20)
            lx, ly = x + ex, y + ey
            rx, ry = x + int(w * 0.5), y + ey

            left_eye = gray[ly:ly + eh, lx:lx + ew]
            right_eye = gray[ry:ry + eh, rx:rx + ew]

            # Detect pupils
            def pupil_centroid(eye_roi):
                if eye_roi.size == 0:
                    return (None, None, None)
                eq = cv2.equalizeHist(eye_roi)
                blur = cv2.GaussianBlur(eq, (7, 7), 0)
                _, th = cv2.threshold(blur, 50, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    return (None, None, None)
                cnt = max(contours, key=cv2.contourArea)
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                if r < 2:
                    return (None, None, None)
                return (int(cx), int(cy), th)

            lc_x, lc_y, lth = pupil_centroid(left_eye)
            rc_x, rc_y, rth = pupil_centroid(right_eye)

            pupil_positions = []
            if lc_x is not None:
                pupil_positions.append((lc_x + lx, lc_y + ly))
            if rc_x is not None:
                pupil_positions.append((rc_x + rx, rc_y + ry))

            if pupil_positions:
                avg_px = int(np.mean([p[0] for p in pupil_positions]))
                avg_py = int(np.mean([p[1] for p in pupil_positions]))
                gaze_x = np.interp(avg_px, [0, frame.shape[1]], [0, screen_w])
                gaze_y = np.interp(avg_py, [0, frame.shape[0]], [0, screen_h])
                smooth_queue.append((gaze_x, gaze_y))
                avg_x = int(np.mean([p[0] for p in smooth_queue]))
                avg_y = int(np.mean([p[1] for p in smooth_queue]))
                pyautogui.moveTo(avg_x, avg_y, _pause=False)
                gaze_point = (avg_x, avg_y)

            # Blink detection
            def dark_ratio_eye(eye_roi):
                if eye_roi.size == 0:
                    return None
                blur = cv2.GaussianBlur(eye_roi, (7, 7), 0)
                th = cv2.adaptiveThreshold(blur, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
                return cv2.countNonZero(th) / (eye_roi.shape[0] * eye_roi.shape[1])

            lr = dark_ratio_eye(left_eye)
            rr = dark_ratio_eye(right_eye)
            if lr is not None and rr is not None:
                avg_ratio = (lr + rr) / 2.0
                if avg_ratio < blink_threshold:
                    current_time = time.time()
                    if current_time - last_click_time > CLICK_COOLDOWN:
                        pyautogui.click()
                        last_click_time = current_time

        # Handle EEG actions
        try:
            while True:
                action = action_queue.get_nowait()
                if action[0] == 'press':
                    keyname, duration = action[1], action[2]
                    if keyname not in active_keys:
                        pyautogui.keyDown(keyname)
                        active_keys[keyname] = time.time() + duration
                        print(f"[EEG] keyDown {keyname} for {duration}s")
        except queue.Empty:
            pass

        # Release keys when done
        now = time.time()
        for k in list(active_keys.keys()):
            if now >= active_keys[k]:
                pyautogui.keyUp(k)
                del active_keys[k]

        # Show window
        cv2.imshow("Hybrid Eye+EEG Control", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()
    for k in list(active_keys.keys()):
        pyautogui.keyUp(k)
    print("Eye control stopped.")


# ======================================================
#                     MAIN FUNCTION
# ======================================================

def main():
    eeg_thread = threading.Thread(target=eeg_worker,
                                  args=(action_queue, stop_event, COM_PORT, BAUD_RATE),
                                  daemon=True)
    eeg_thread.start()

    try:
        start_eye_control(action_queue, stop_event)
    except KeyboardInterrupt:
        stop_event.set()

    stop_event.set()
    print("[MAIN] Waiting for EEG thread to finish...")
    eeg_thread.join(timeout=2.0)
    print("[MAIN] Exited.")


if __name__ == "__main__":
    main()