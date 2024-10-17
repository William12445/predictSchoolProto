import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Variables for click timing and scrolling state
last_click_time = 0
double_click_threshold = 0.5  # 0.5 seconds for double-click detection
scrolling_enabled = False
scroll_speed = 50  # Speed of scrolling
prev_left_wrist_y = None
click_enabled = True  # To prevent multiple clicks until hand is back to normal

# Function to calculate the Euclidean distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to check if the hand is making a pinch gesture
def is_pinch(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    # Calculate distances for pinch gestures
    thumb_index_distance = calculate_distance(
        thumb_tip.x, thumb_tip.y,
        index_finger_tip.x, index_finger_tip.y
    )

    thumb_middle_distance = calculate_distance(
        thumb_tip.x, thumb_tip.y,
        middle_finger_tip.x, middle_finger_tip.y
    )

    # Define threshold for pinch gesture
    pinch_threshold = 0.05
    return thumb_index_distance < pinch_threshold or thumb_middle_distance < pinch_threshold

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize the MediaPipe Hands module
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the image for correct mirror effect and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find hand landmarks
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = hand_info.classification[0].label  # 'Left' or 'Right'

                # Detect left hand for scrolling (pinch gesture)
                if hand_label == 'Left':
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

                    # Check for pinch gesture with the left hand
                    if is_pinch(hand_landmarks.landmark):
                        scrolling_enabled = True
                        print("Pinch detected - Scrolling enabled")
                        
                        # Use the Y-coordinate of the wrist to control scrolling
                        left_wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        left_wrist_y = int(left_wrist.y * screen_height)

                        # Implement scroll logic based on wrist movement
                        if prev_left_wrist_y is not None:
                            if left_wrist_y < prev_left_wrist_y - 20:  # Move hand up to scroll up
                                pyautogui.scroll(scroll_speed)
                            elif left_wrist_y > prev_left_wrist_y + 20:  # Move hand down to scroll down
                                pyautogui.scroll(-scroll_speed)

                        # Update previous wrist position
                        prev_left_wrist_y = left_wrist_y
                    else:
                        scrolling_enabled = False
                        prev_left_wrist_y = None  # Reset previous wrist position when not pinching

                # Detect right hand for cursor control and clicks
                elif hand_label == 'Right':
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

                    # Get the thumb and index/middle finger tip coordinates for clicks
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                    # Calculate the center of the palm (between wrist and thumb base)
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    palm_center_x = (wrist.x + thumb_tip.x) / 2
                    palm_center_y = (wrist.y + thumb_tip.y) / 2

                    # Convert the normalized palm center coordinates to screen coordinates
                    cursor_x = int(palm_center_x * screen_width)
                    cursor_y = int(palm_center_y * screen_height)

                    # Move the cursor based on the palm center
                    pyautogui.moveTo(cursor_x, cursor_y)

                    # Calculate distances for pinch gestures
                    left_click_distance = calculate_distance(
                        thumb_tip.x, thumb_tip.y,
                        index_finger_tip.x, index_finger_tip.y
                    )

                    right_click_distance = calculate_distance(
                        thumb_tip.x, thumb_tip.y,
                        middle_finger_tip.x, middle_finger_tip.y
                    )

                    # Left click: Pinch gesture with thumb and index finger
                    if left_click_distance < 0.05:  # Adjust threshold as needed
                        current_time = time.time()
                        if click_enabled:  # Only allow clicking if the hand is in pinch form
                            if current_time - last_click_time < double_click_threshold:
                                pyautogui.doubleClick()  # Perform double-click
                                print("Double-click detected")
                            else:
                                pyautogui.click()  # Perform left click
                                print("Left click detected")
                            last_click_time = current_time  # Update last click time
                            click_enabled = False  # Disable further clicks until hand returns to normal

                    # Right click: Pinch gesture with thumb and middle finger
                    elif right_click_distance < 0.05:  # Adjust threshold as needed
                        if click_enabled:  # Only allow right click if the hand is in pinch form
                            pyautogui.rightClick()  # Perform right click
                            print("Right click detected")
                            click_enabled = False  # Disable further clicks until hand returns to normal

                    # Reset click_enabled when hand is back to normal
                    if not (left_click_distance < 0.05 or right_click_distance < 0.05):
                        click_enabled = True  # Enable clicking again when hand is normal

        # Show the video feed with hand tracking
        cv2.imshow('Hand Tracking - Cursor and Scrolling', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
