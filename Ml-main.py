import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image_path = "handwritten_digit.jpg"  # Path to your handwritten digit image
image = Image.open(image_path)

# Preprocess the image
image = image.convert("L")  # Convert to grayscale
image = image.resize((8, 8))  # Resize to 8x8 pixels
image_array = np.array(image)

# Flatten the image array
image_flattened = image_array.flatten()

# Normalize pixel values
image_normalized = image_flattened / 255.0

# Load trained model
# Assume model is already trained and saved as 'trained_model.pkl'
import pickle

with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make prediction
predicted_digit = model.predict([image_normalized])[0]

# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(image_array, cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("Original Image")

# Plot the predicted digit
plt.subplot(1, 2, 2)
plt.text(0.5, 0.5, str(predicted_digit), fontsize=50, ha='center')
plt.title("Predicted Digit")

plt.show()
