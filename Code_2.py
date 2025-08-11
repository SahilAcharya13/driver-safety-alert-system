import cv2
import mediapipe as mp
import numpy as np
import playsound
from threading import Thread
import argparse
def sound_alarm(path):
    playsound.playsound("Alert.wav")

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Eye and facial landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14]
NOSE = 1


# EAR (Eye Aspect Ratio) function
def eye_aspect_ratio(landmarks, eye):
    A = np.linalg.norm(landmarks[eye[1]] - landmarks[eye[5]])
    B = np.linalg.norm(landmarks[eye[2]] - landmarks[eye[4]])
    C = np.linalg.norm(landmarks[eye[0]] - landmarks[eye[3]])
    ear = (A + B) / (2.0 * C)
    return ear


# Mouth opening function
def mouth_open_ratio(landmarks):
    return np.linalg.norm(landmarks[MOUTH[0]] - landmarks[MOUTH[1]])


# Head movement detection (left/right/center)
def detect_head_movement(nose, frame_width):
    center = frame_width // 2
    if nose[0] < center - 60:
        return -1
    elif nose[0] > center + 60:
        return 1
    else:
        return 0


# Thresholds
EYE_AR_THRESH = 0.25  # Less sensitive
MOUTH_THRESH = 20  # Adjust if needed
DROWSY_FRAMES = 0
YAWN_FRAMES = 0
saying = False
alarm_status2 = False
alarm_status = False

# Start webcam
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if unavailable
# CLI arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="Webcam index")
ap.add_argument("-a", "--alarm", type=str, default="Alert.wav", help="Alarm sound path")
args = vars(ap.parse_args())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([[p.x * w, p.y * h] for p in face_landmarks.landmark])

            # EAR and mouth ratio
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            mouth_ratio = mouth_open_ratio(landmarks)

            # EAR display
            cv2.putText(frame, f"", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Drowsiness detection
            if avg_ear < EYE_AR_THRESH:
                DROWSY_FRAMES += 1
            else:
                DROWSY_FRAMES = 0

            if DROWSY_FRAMES > fps * 2:  # 2 seconds at current FPS
                cv2.putText(frame, "DROWSINESS ALERT!", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                if not alarm_status:
                    alarm_status = True
                    if not alarm_status:
                        alarm_status = True
                        try:
                            Thread(target=sound_alarm, args=(args["alarm"],)).start()
                        except Exception as e:
                            print("Error playing sound:", e)
                    else:
                        COUNTER = 0
                        alarm_status = False
            # Yawn detection
            if mouth_ratio > MOUTH_THRESH:
                YAWN_FRAMES += 1
            else:
                YAWN_FRAMES = 0

            if YAWN_FRAMES > fps * 1.5:  # 1.5 seconds mouth open
                cv2.putText(frame, "YAWN ALERT!", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                if not alarm_status2 and not saying:
                    alarm_status2 = True
                    if not alarm_status:
                        alarm_status = True
                        try:
                            Thread(target=sound_alarm, args=(args["alarm"],)).start()
                        except Exception as e:
                            print("Error playing sound:", e)

            else:
                COUNTER = 0
                alarm_status = False
            # Head movement
            nose_x, nose_y = landmarks[NOSE]
            head_dir = detect_head_movement((nose_x, nose_y), w)

            if head_dir == -1:
                cv2.putText(frame, "Looking Right", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            elif head_dir == 1:
                cv2.putText(frame, "Looking Left", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Looking CENTER", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Drowsiness & Yawn Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
