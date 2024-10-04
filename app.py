import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match model input
    image = np.array(image) / 255.0    # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Upload model file
uploaded_model = st.file_uploader("Upload your Keras model (.h5)", type="h5")

if uploaded_model is not None:
    try:
        model = load_model(uploaded_model)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    # Optionally load a default model if not uploaded
    # Uncomment and specify the correct path if you have a default model
    # model = load_model('your_default_model.h5')

    st.warning("Please upload your Keras model to proceed.")

# Image upload for prediction
uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])

if uploaded_image is not None and 'model' in locals():
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)

    # Display prediction result
    st.write(f"Predicted class: {predicted_class[0]}")
