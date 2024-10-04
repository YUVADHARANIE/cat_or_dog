import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Check current working directory
st.write("Current Working Directory: ", os.getcwd())

# Load the model
try:
    model = load_model("keras_model.h5", compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load the labels
try:
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except Exception as e:
    st.error(f"Error loading labels: {e}")

# Function to preprocess the image
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return np.expand_dims(normalized_image_array, axis=0)  # Expand dims to fit model input

# Streamlit UI
st.title("Image Classification with Keras")
st.write("Upload an image to classify.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    data = preprocess_image(image)

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display prediction and confidence score
    st.write(f"**Class:** {class_name[2:]}")  # Adjusting if necessary to skip unwanted characters
    st.write(f"**Confidence Score:** {confidence_score:.2f}")
