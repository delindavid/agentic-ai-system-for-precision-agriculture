import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
from PIL import Image

IMG_SIZE = 128

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

# Load class names
with open("classes.pkl", "rb") as f:
    class_names = pickle.load(f)

st.title("🌾 Rice Classification App")
st.write("Upload a rice image to predict its class")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image for prediction
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.subheader("Prediction:")
    st.success(f"{class_names[class_index]}")
    st.write(f"Confidence: {confidence:.2f}%")