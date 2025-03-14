import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

r = random.randint(2,349)


def preprocess_image(image_path, image_size=(28, 28)):

    image = Image.open(image_path).convert("L")
    image = image.resize(image_size)
    image_array = np.array(image).astype(np.float32)
    image_array /= 255.0
    image_flatten = image_array.flatten().reshape(-1, 1)
    return image_flatten

with open("model\\nn_mnist_model.pkl", "rb") as f:
    nn_mnist = pickle.load(f)

test_image_path = f"data\\image\\img_{r}.jpg"
test_image_vector = preprocess_image(test_image_path)

predicted_digit = nn_mnist.predict(test_image_vector)
image = Image.open(test_image_path).convert("L")

plt.imshow(image, cmap='gray')
plt.title(f"Predicted Digit: {predicted_digit[0]}")
plt.axis('off')
plt.show()
