import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# Title of the Streamlit app
st.title("Teachable Machine Model with Streamlit")

# Load the model from your repository
MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"

# Load class labels from the labels.txt file
def load_labels(label_file):
    try:
        with open(label_file, "r") as f:
            labels = f.read().splitlines()
        return labels
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return []

class_names = load_labels(LABELS_PATH)

# Function to load the model
@st.cache_resource
def load_keras_model():
    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model once the app starts
model = load_keras_model()
if model:
    st.success("Model loaded successfully!")

# Upload an image for prediction
uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "png", "jpeg"])

if uploaded_image is not None and model is not None:
    # Load and display the uploaded image
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Resize and preprocess the image
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array and normalize it
    image_array = np.asarray(img)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make predictions
    if st.button("Predict"):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Display the results
        st.write(f"Predicted Class: {class_name.strip()}")
        st.write(f"Confidence Score: {confidence_score:.2f}")

