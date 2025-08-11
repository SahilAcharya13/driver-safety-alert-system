import cv2
import time
import numpy as np
import mediapipe as mp
from threading import Thread
import argparse
import playsound

def sound_alarm(path):
    playsound.playsound(path)

def eye_aspect_ratio(landmarks, left_indices, right_indices):
    def compute_ear(eye_points):
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        return (A + B) / (2.0 * C)

    left_eye = np.array([landmarks[i] for i in left_indices])
    right_eye = np.array([landmarks[i] for i in right_indices])
    left_ear = compute_ear(left_eye)
    right_ear = compute_ear(right_eye)
    return (left_ear + right_ear) / 2.0, left_eye, right_eye

def lip_distance(landmarks, top_indices, bottom_indices):
    top_lip = np.array([landmarks[i] for i in top_indices])
    bottom_lip = np.array([landmarks[i] for i in bottom_indices])
    top_mean = np.mean(top_lip, axis=0)
    bottom_mean = np.mean(bottom_lip, axis=0)
    return abs(top_mean[1] - bottom_mean[1])

# CLI arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="Webcam index")
ap.add_argument("-a", "--alarm", type=str, default="Alert.wav", help="Alarm sound path")
args = vars(ap.parse_args())

# Constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 25
COUNTER = 0
alarm_status = False
alarm_status2 = False
saying = False

# Indices for facial landmarks (MediaPipe uses 468 landmarks)
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
TOP_LIP_IDX = [13, 312, 308, 324]
BOTTOM_LIP_IDX = [14, 87, 317, 402]

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Start webcam
cap = cv2.VideoCapture(args["webcam"])
time.sleep(1.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

            ear, left_eye_pts, right_eye_pts = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, RIGHT_EYE_IDX)
            distance = lip_distance(landmarks, TOP_LIP_IDX, BOTTOM_LIP_IDX)

            for eye in [left_eye_pts, right_eye_pts]:
                cv2.polylines(frame, [np.array(eye, dtype=np.int32)], True, (0, 255, 0), 1)

            mouth = [landmarks[i] for i in TOP_LIP_IDX + BOTTOM_LIP_IDX]
            cv2.polylines(frame, [np.array(mouth, dtype=np.int32)], True, (255, 0, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not alarm_status:
                        alarm_status = True
                        Thread(target=sound_alarm, args=(args["alarm"],), daemon=True).start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                alarm_status = False

            if distance > YAWN_THRESH:
                cv2.putText(frame, "YAWN ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not alarm_status2 and not saying:
                    alarm_status2 = True
                    Thread(target=sound_alarm, args=(args["alarm"],), daemon=True).start()
            else:
                alarm_status2 = False

            cv2.putText(frame, f"EAR: {ear:.2f}", (400, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"YAWN: {distance:.2f}", (400, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
