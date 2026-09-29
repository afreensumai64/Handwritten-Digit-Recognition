import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from pathlib import Path

# Load MNIST test data
(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_test = X_test.astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

# Load trained model
model_path = Path(__file__).resolve().parents[1] / "models" / "handwritten_digit_cnn.keras"
model = tf.keras.models.load_model(model_path)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
print("Test Accuracy (%):", accuracy * 100)

# Generate predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)