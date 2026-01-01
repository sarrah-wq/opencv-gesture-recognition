import numpy as np
import pandas as pd
from sklearn.svm import SVC
import joblib

# -------------------------------
# Load dataset (local files)
# -------------------------------
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

X_train = train.iloc[:, 1:].values / 255.0
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values / 255.0
y_test = test.iloc[:, 0].values

# -------------------------------
# Train SVM model
# -------------------------------
# RBF kernel works best for images
svm = SVC(
    kernel='rbf',
    C=10,
    gamma='scale'
)

svm.fit(X_train, y_train)

# -------------------------------
# Save model
# -------------------------------
joblib.dump(svm, 'svm_gesture_model.pkl')
print("SVM model saved as svm_gesture_model.pkl")

# -------------------------------
# Evaluate
# -------------------------------
accuracy = svm.score(X_test, y_test)
print(f"SVM Test Accuracy: {accuracy:.4f}")
