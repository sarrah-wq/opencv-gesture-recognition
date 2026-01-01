import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# -------------------------------
# Load dataset
# -------------------------------
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

X_train = train.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_test = test.iloc[:, 0].values

# One-hot encoding
y_train = to_categorical(y_train, num_classes=25)
y_test = to_categorical(y_test, num_classes=25)

# -------------------------------
# Build CNN
# -------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(25, activation='softmax')
])

# -------------------------------
# Compile model
# -------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Train
# -------------------------------
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_data=(X_test, y_test)
)

# -------------------------------
# Save model
# -------------------------------
model.save('cnn_gesture_model.h5')
print("CNN model saved as cnn_gesture_model.h5")
