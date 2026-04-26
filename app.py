import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Potato Disease Detection", layout="centered")

st.title("🥔 Potato Leaf Disease Detection")
st.write("Upload a potato leaf image to detect disease")

# ==============================
# LOAD MODEL (IMPORTANT)
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# ==============================
# CLASS NAMES (CHANGE IF NEEDED)
# ==============================
class_names = ["Early Blight", "Late Blight", "Healthy"]

# ==============================
# IMAGE PREPROCESS FUNCTION
# ==============================
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ==============================
    # PREDICTION
    # ==============================
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # ==============================
    # RESULT
    # ==============================
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}")