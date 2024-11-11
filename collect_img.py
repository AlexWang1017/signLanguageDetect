import cv2
import mediapipe as mp
import os
import numpy as np

# Set up directories for saving images
data_dir = '/mnt/data/dataset/'
os.makedirs(data_dir, exist_ok=True)
class_names = ['5', 'hello']

# Create subdirectories for each class
for class_name in class_names:
    class_path = os.path.join(data_dir, class_name)
    os.makedirs(class_path, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Parameters for collecting images
num_images = 200  # Number of images per class
image_size = (64, 64)  # Resize images to 64x64 pixels

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    for class_name in class_names:
        print(f"Collecting images for class: {class_name}")
        count = 0

        while count < num_images:
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
            frame.flags.writeable = True
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw instructions on the frame
            cv2.putText(frame, f"Class: {class_name}, Image: {count + 1}/{num_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Collecting Images', frame)

            # Wait for user to press 's' to save the image
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and results.multi_hand_landmarks:
                # Resize and save the image
                resized_frame = cv2.resize(frame, image_size)
                save_path = os.path.join(data_dir, class_name, f"{count}.jpg")
                cv2.imwrite(save_path, resized_frame)
                count += 1

            # Exit if 'q' is pressed
            elif key == ord('q'):
                break

        if count < num_images:
            print(f"Collection for class {class_name} was interrupted.")
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
