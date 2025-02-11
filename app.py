import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
# here
import os
import requests
from tensorflow.keras.models import load_model

MODEL_URL = "https://drive.google.com/file/d/1QK7J27P6nFEseUzWtzBA8ocyKuIEhRSn/view?usp=drive_link"
MODEL_PATH = "bio.h5"

def download_model(url=MODEL_URL, dest=MODEL_PATH):
    if not os.path.exists(dest):  # Download only if not already present
        st.write("Downloading model... Please wait.")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.write("Download complete.")

download_model()

@st.cache_resource
def load_model():
    return load_model(MODEL_PATH)

model = load_model()
#here
# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("bio.h5")

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to match model input
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Expand dimensions for model input
    return image

# Streamlit UI
st.title("Waste Classification App")
st.write("Upload an image to classify it as **Biodegradable** or **Non-Biodegradable**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]  # Assuming binary classification

    # Display result
    label = "Biodegradable" if prediction > 0.5 else "Non-Biodegradable"
    st.write(f"### Prediction: **{label}**")
