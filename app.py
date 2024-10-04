import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your pre-trained model
# Ensure the model file is in the same directory or provide the correct path
model = load_model('your_model.h5')  # Change 'your_model.h5' to your model file

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize to the expected input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to make predictions
def predict(img):
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    return 'Dog' if predictions[0][0] > 0.5 else 'Cat'

# Streamlit UI
st.title('Cat or Dog Classifier')
st.write("Upload an image of a cat or a dog to see the prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    label = predict(img)
    st.write(f'This image is a: **{label}**')
