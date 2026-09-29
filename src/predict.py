import tensorflow as tf
import numpy as np
from pathlib import Path

# Load trained model
model_path = Path(__file__).resolve().parents[1] / "models" / "handwritten_digit_cnn.keras"
model = tf.keras.models.load_model(model_path)

# Load MNIST test data for demonstration
(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Select one image
image = X_test[0]

# Preprocess image
image_processed = image.astype("float32") / 255.0
image_processed = image_processed.reshape(1, 28, 28, 1)

# Predict digit
prediction = model.predict(image_processed)
predicted_digit = np.argmax(prediction)

print("Predicted digit:", predicted_digit)
print("Actual digit:", y_test[0])