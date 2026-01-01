import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

# -------------------------------
# Load the dataset (local files)
# -------------------------------
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

# Split features and labels
X_train = train.iloc[:, 1:].values / 255.0  # normalize pixels
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values / 255.0
y_test = test.iloc[:, 0].values

# -------------------------------
# Initialize and train KNN model
# -------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# -------------------------------
# Save the trained model
# -------------------------------
joblib.dump(knn, 'knn_gesture_model.pkl')
print("KNN model trained and saved as 'knn_gesture_model.pkl'")

# -------------------------------
# Optional: Evaluate accuracy
# -------------------------------
y_pred = knn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"KNN Test Accuracy: {accuracy:.4f}")