import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('ML_app/cats_vs_dogs_model.h5')

def predict_image(image_path):
    image = Image.open(image_path).resize((150, 150))
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Expand dims
    prediction = model.predict(image_array)
    return "Dog" if prediction[0] > 0.5 else "Cat"
