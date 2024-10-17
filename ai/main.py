import subprocess
import sys
import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Function to install modules
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required modules
required_modules = [
    ('opencv-python', 'cv2'),
    ('mediapipe', 'mediapipe'),
    ('pyautogui', 'pyautogui')
]

# Check if modules are installed, if not install them
for package, module in required_modules:
    try:
        __import__(module)
    except ImportError:
        print(f"{module} not found, installing {package}...")
        install(package)

# Initialize Mediapipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Capture webcam feed
cap = cv2.VideoCapture(0)

# Set the webcam resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Screen resolution for scaling hand movement to screen size
screen_width, screen_height = pyautogui.size()

# Variables to store previous hand positions and state
last_fist_x = None
last_fist_y = None
last_fist_movement_time = time.time()  # Track the time of the last fist movement
last_click_time = 0  # Track the time of the last click
click_made = False  # Track whether a click has been made

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

# Function to check if middle finger tip touches the thumb tip (for right hand click)
def is_middle_finger_touching_thumb(landmarks):
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]

    # Calculate the distance between the middle finger tip and the thumb tip
    distance = calculate_distance(middle_tip.x, middle_tip.y, thumb_tip.x, thumb_tip.y)

    # Define a threshold to decide when the middle finger tip is "touching" the thumb tip
    return distance < 0.05  # Adjust threshold if necessary

# Function to debounce clicks
def debounce_click(min_interval=0.5):
    global last_click_time
    current_time = time.time()
    if current_time - last_click_time > min_interval:
        pyautogui.click()
        last_click_time = current_time

# Function to detect if the left hand is making a fist
def is_left_fist(landmarks):
    # Check if all fingers except the thumb are folded into a fist
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]

    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_cmc = landmarks[mp_hands.HandLandmark.THUMB_CMC]

    # Calculate the distance between thumb tip and thumb base (to make sure thumb is open)
    thumb_open = calculate_distance(thumb_tip.x, thumb_tip.y, thumb_cmc.x, thumb_cmc.y) > 0.1

    # Check if other fingers are closed (i.e., tips near the palm)
    for tip in finger_tips:
        finger_tip = landmarks[tip]
        finger_base = landmarks[tip - 2]  # Compare finger tip with the base (MCP joint)
        if finger_tip.y < finger_base.y:  # If the tip is higher than base, it's not a fist
            return False

    return thumb_open  # The hand is a fist only if the fingers are folded, and the thumb is open

# Function to handle left hand scrolling
def handle_left_hand_scroll(landmarks):
    global last_fist_x, last_fist_y, last_fist_movement_time

    # Check if a fist is made
    if is_left_fist(landmarks):
        # Get the position of the fist
        fist = landmarks[mp_hands.HandLandmark.WRIST]
        fist_x = int(fist.x * screen_width)
        fist_y = int(fist.y * screen_height)

        if last_fist_x is not None and last_fist_y is not None:
            # Calculate movement direction and speed
            delta_x = fist_x - last_fist_x
            delta_y = fist_y - last_fist_y
            current_time = time.time()
            time_elapsed = current_time - last_fist_movement_time

            # Scale the scroll amount based on movement and time elapsed
            # scroll_amount_x = delta_x * 0.1 / time_elapsed
            scroll_amount_y = delta_y * 0.1 / time_elapsed

            # Smooth scrolling based on movement direction
            if abs(delta_x) > abs(delta_y):
                if delta_y > 0:
                    pyautogui.scroll(-scroll_amount_y)  # Smooth scroll down
                else:
                    pyautogui.scroll(scroll_amount_y)  # Smooth scroll up

        last_fist_x = fist_x
        last_fist_y = fist_y
        last_fist_movement_time = time.time()
    else:
        last_fist_x = None
        last_fist_y = None

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    # Flip the frame for natural movement
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(rgb_frame)

    # Get image dimensions
    frame_height, frame_width, _ = frame.shape

    # Check if any hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = hand_info.classification[0].label  # 'Left' or 'Right'

            if hand_label == 'Right':  # Right hand controls cursor and clicks
                # Use the thumb tip to control cursor movement
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_tip_x = int(thumb_tip.x * screen_width)
                thumb_tip_y = int(thumb_tip.y * screen_height)

                # Ensure the cursor stays within screen boundaries
                thumb_tip_x = max(0, min(screen_width - 1, thumb_tip_x))
                thumb_tip_y = max(0, min(screen_height - 1, thumb_tip_y))

                # Move cursor
                pyautogui.moveTo(thumb_tip_x, thumb_tip_y)

                # Check if middle finger tip touches thumb tip and perform click
                if is_middle_finger_touching_thumb(hand_landmarks.landmark):
                    if not click_made:
                        # Change color to indicate click
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
                        debounce_click()  # Click when touching
                        click_made = True
                else:
                    click_made = False

            if hand_label == 'Left':  # Left hand controls scrolling
                # Change color to green if fist is formed
                if is_left_fist(hand_landmarks.landmark):
                    color = (0, 255, 0)  # Green
                else:
                    color = (0, 0, 255)  # Red

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2))
                handle_left_hand_scroll(hand_landmarks.landmark)

    # Display the frame with landmarks
    cv2.imshow('Hand Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
