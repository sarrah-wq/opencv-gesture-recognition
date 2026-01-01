import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('cnn_gesture_model.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    x0, y0, w, h = 100, 100, 200, 200
    roi = frame[y0:y0+h, x0:x0+w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    img = cv2.resize(thresh, (28,28))
    img = img.reshape(1,28,28,1) / 255.0

    pred = model.predict(img, verbose=0)
    gesture = np.argmax(pred)

    cv2.rectangle(frame, (x0,y0), (x0+w,y0+h), (0,255,0), 2)
    cv2.putText(frame, f'Gesture: {gesture}', (x0,y0-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("CNN Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
