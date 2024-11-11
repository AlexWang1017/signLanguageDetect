import os
import numpy as np
import cv2

# Set up directories for loading images
data_dir = '/mnt/data/dataset/'
class_names = ['5', 'hello']
image_size = (64, 64)  # Size of images

# Prepare lists to store images and labels
x_data = []
y_data = []

# Load images and labels
for label, class_name in enumerate(class_names):
    class_path = os.path.join(data_dir, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            # Convert image to RGB to be consistent with MediaPipe processing
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)
            x_data.append(img)
            y_data.append(label)

# Convert lists to numpy arrays
x_data = np.array(x_data, dtype='float32') / 255.0  # Normalize images to [0, 1]
y_data = np.array(y_data, dtype='int')

# Save the dataset
np.save(os.path.join(data_dir, 'x_data.npy'), x_data)
np.save(os.path.join(data_dir, 'y_data.npy'), y_data)

print(f"Dataset created with {len(x_data)} samples.")
