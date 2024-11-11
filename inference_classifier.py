import numpy as np
import cv2
import os
import tensorflow as tf
from collections import deque
import mediapipe as mp

# Load the trained model
data_dir = '/mnt/data/dataset/'
model_path = os.path.join(data_dir, 'gesture_classifier.h5')
model = tf.keras.models.load_model(model_path)

# Set up class names
class_names = ['5', 'hello']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Parameters for inference
image_size = (64, 64)  # Resize images to 64x64 pixels
frame_buffer = deque(maxlen=5)  # Buffer to hold the last 5 frames
previous_landmarks = None  # To store previous frame landmarks

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Flip the image horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = hands.process(image_rgb)

        # Draw hand annotations on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks as (x, y) tuples
                current_landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

                # Compare with previous landmarks to determine if there is movement
                if previous_landmarks is not None:
                    movement_detected = any(
                        np.linalg.norm(np.array(current) - np.array(previous)) > 0.02
                        for current, previous in zip(current_landmarks, previous_landmarks)
                    )
                else:
                    movement_detected = False

                # Update previous landmarks
                previous_landmarks = current_landmarks

                # Determine the gesture based on movement
                if movement_detected:
                    predicted_label = 'hello'
                else:
                    predicted_label = '5'

                # Display the prediction
                cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Gesture Recognition', frame)

        # Exit if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()