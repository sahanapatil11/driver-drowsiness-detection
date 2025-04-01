import cv2
import numpy as np
import dlib
from imutils import face_utils
import time
from playsound import playsound  # To play alert sound

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Status counters
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Function to compute distance between two points
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# Function to check eye closure
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2  # Open eyes
    elif ratio > 0.21 and ratio <= 0.25:
        return 1  # Drowsy
    else:
        return 0  # Sleeping

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))  # Resize for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Detect eye status
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], 
                             landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], 
                              landmarks[47], landmarks[46], landmarks[45])
        
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                playsound("cyber-alarms-synthesized-116358.mp3")  # Play alert sound

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)

        # Display status
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw landmarks
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Drowsiness Detector", frame)

    # Exit on pressing ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
