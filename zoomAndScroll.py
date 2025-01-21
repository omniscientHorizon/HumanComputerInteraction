import cv2
import pyautogui
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

scrolling_enabled = True
zooming_enabled = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks for index finger and thumb
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            

            # Calculate distance between index finger tip and thumb tip
            distance = math.sqrt(
                (index_finger_tip.x - thumb_tip.x) ** 2 +
                (index_finger_tip.y - thumb_tip.y) ** 2
            )

            # Detect closed fist 
            if distance < 0.05:  
                scrolling_enabled = False
                zooming_enabled = False
                hand_gesture = "fist (stop scrolling)"
            else:
                scrolling_enabled = True
                zooming_enabled = True
            
                if index_finger_tip.y < thumb_tip.y:
                    hand_gesture = "pointing up"
                elif index_finger_tip.y > thumb_tip.y:
                    hand_gesture = "pointing down"
                elif pinky_tip.y < thumb_tip.y:
                    hand_gesture = "zoom in"
                elif pinky_tip.y > thumb_tip.y:
                    hand_gesture = "zoom out"
                else:
                    hand_gesture = "other"

            
            if scrolling_enabled:
                if hand_gesture == "pointing up":
                    pyautogui.scroll(2) 
                elif hand_gesture == "pointing down":
                    pyautogui.scroll(-2) 
            
            if zooming_enabled:
                if hand_gesture == "zoom in":
                    pyautogui.hotkey('command', '=')  
                elif hand_gesture == "zoom out":
                    pyautogui.hotkey('command', '-') 

    cv2.imshow('Hand Gesture', frame)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
