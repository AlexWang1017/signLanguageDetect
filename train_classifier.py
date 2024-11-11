# Import necessary libraries
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os

# Load data
data_dir = '/mnt/data/dataset/'
x_data = np.load(os.path.join(data_dir, 'x_data.npy'))
y_data = np.load(os.path.join(data_dir, 'y_data.npy'))

# Check and reshape data to ensure it has the right dimensions
num_samples, height, width, channels = x_data.shape
num_frames = 5  # Assuming each sequence contains 5 frames

# Reshape x_data to have shape (num_sequences, num_frames, height, width, channels)
x_data = x_data[:num_samples // num_frames * num_frames]  # Make num_samples divisible by num_frames
x_data = x_data.reshape((num_samples // num_frames, num_frames, height, width, channels))
y_data = y_data[:num_samples // num_frames]  # Adjust y_data length accordingly

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Define the CNN-RNN model
model = Sequential()

# TimeDistributed CNN
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))

# LSTM Layer
model.add(LSTM(50, return_sequences=False))

# Fully connected layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # Assuming 2 classes: '5' and 'hello'

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 20
batch_size = 16
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

# Save the model
model.save(os.path.join(data_dir, 'gesture_classifier.h5'))

# Evaluate the model
evaluation = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")
 
