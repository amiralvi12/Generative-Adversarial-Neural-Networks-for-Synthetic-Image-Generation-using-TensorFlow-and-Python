import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("model file/final_model.h5")

# Load test images (augmented)
augmented_data = np.load("Augmented_Data/augmented_images.npy")
augmented_data = augmented_data / 255.0  # Normalize

# Make Predictions
predictions = model.predict(augmented_data[:5])

# Show Results
for i in range(5):
    plt.imshow(augmented_data[i].reshape(28,28), cmap="gray")
    plt.title(f"Predicted Label: {np.argmax(predictions[i])}")
    plt.axis("off")
    plt.show()
