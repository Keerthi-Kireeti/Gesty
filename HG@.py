import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Get volume range
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]
vol_step = 5  # Step size for volume increase/decrease

def get_finger_status(landmarks):
    """Detects hand gestures for volume control."""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    thumb_mcp = landmarks[2]  # Thumb base

    # Thumbs up: Thumb extended, others closed
    thumbs_up = thumb_tip.y < thumb_mcp.y and all(
        landmarks[tip_idx].y > landmarks[tip_idx - 2].y for tip_idx in [8, 12, 16, 20]
    )

    # Thumbs down: Thumb pointing down, others closed
    thumbs_down = thumb_tip.y > thumb_mcp.y and all(
        landmarks[tip_idx].y > landmarks[tip_idx - 2].y for tip_idx in [8, 12, 16, 20]
    )

    # Fist: All fingers closed
    fist = all(landmarks[tip_idx].y > landmarks[tip_idx - 2].y for tip_idx in [4, 8, 12, 16, 20])

    return thumbs_up, thumbs_down, fist

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            thumbs_up, thumbs_down, fist = get_finger_status(landmarks)

            if thumbs_up:
                current_vol = volume.GetMasterVolumeLevel()
                new_vol = min(current_vol + vol_step, max_vol)
                volume.SetMasterVolumeLevel(new_vol, None)
                print("Volume Increased")

            elif thumbs_down:
                current_vol = volume.GetMasterVolumeLevel()
                new_vol = max(current_vol - vol_step, min_vol)
                volume.SetMasterVolumeLevel(new_vol, None)
                print("Volume Decreased")

            elif fist:
                mute_status = volume.GetMute()
                volume.SetMute(not mute_status, None)
                print("Mute Toggled")

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
