import cv2
import numpy as np
import joblib

# ---------------------------------------
# Load trained KNN model
# ---------------------------------------
# This model was trained on 28x28 images
# Each image = 784 features (28*28 pixels)
knn = joblib.load('svm_gesture_model.pkl')

# ---------------------------------------
# Open webcam
# ---------------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror image so movement feels natural
    frame = cv2.flip(frame, 1)

    # ---------------------------------------
    # 1) REGION OF INTEREST (ROI)
    # ---------------------------------------
    # We take a 200x200 square where the hand should be.
    # 200x200 is large enough to capture the whole hand.
    x0, y0, width, height = 100, 100, 200, 200
    roi = frame[y0:y0 + height, x0:x0 + width]

    # Draw ROI box
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 2)

    # ---------------------------------------
    # 2) PREPROCESSING
    # ---------------------------------------

    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Convert to black & white (binary image)
    # Hand becomes white, background becomes black
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    # ---------------------------------------
    # 3) RESIZE TO 28x28
    # ---------------------------------------
    # The model was trained on 28x28 images,
    # so input MUST be 28x28
    img_28 = cv2.resize(thresh, (28, 28))

    # ---------------------------------------
    # 4) FLATTEN + RESHAPE
    # ---------------------------------------
    # 28x28 = 784 pixels
    # KNN expects input shape: (number_of_samples, number_of_features)
    # Here: 1 image → (1, 784)
    img_flat = img_28.flatten().reshape(1, -1)

    # Normalize pixel values (0–1)
    img_flat = img_flat / 255.0

    # ---------------------------------------
    # 5) PREDICTION
    # ---------------------------------------
    prediction = knn.predict(img_flat)[0]

    # ---------------------------------------
    # 6) DISPLAY RESULTS
    # ---------------------------------------
    cv2.putText(
        frame,
        f'Gesture: {prediction}',
        (x0, y0 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show webcam and processed hand
    cv2.imshow("Hand Gesture Recognition", frame)
    cv2.imshow("Processed Hand (28x28)", img_28)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------------------
# Cleanup
# ---------------------------------------
cap.release()
cv2.destroyAllWindows()
