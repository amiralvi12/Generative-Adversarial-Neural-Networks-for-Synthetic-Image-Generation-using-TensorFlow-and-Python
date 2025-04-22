import sys
print(sys.executable)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


generator = tf.keras.models.load_model('generator.keras')


num_images = 16
latent_dim = 128

# Generate random noise vectors
noise = tf.random.normal((num_images, latent_dim))

# Use the generator to predict (generate) images
generated_images = generator.predict(noise)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))

# Plot the generated images
for i in range(num_images):
    row = i // 4
    col = i % 4
    # Squeeze the image dimensions to remove the channel if it's 1 (grayscale)
    img = tf.squeeze(generated_images[i])
    axes[row, col].imshow(img, cmap='gray')  # Use 'gray' colormap for grayscale images
    axes[row, col].axis('off')  # Turn off axis labels and ticks

plt.suptitle(f'Generated Fashion MNIST Images ({num_images} Samples)')
plt.tight_layout()
plt.show()