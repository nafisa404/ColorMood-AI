
import streamlit as st
import cv2
import numpy as np
from src.color_extractor import extract_dominant_colors
from src.model import load_model
from src.config import MODEL_PATH
import joblib

st.title("🎨 ColorMoodAI")

uploaded_file = st.file_uploader("Upload your artwork", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption="Uploaded Artwork")

    st.subheader("🎨 Extracting Colors...")
    colors = extract_dominant_colors(image, k=5)
    st.write("Dominant Colors (BGR):", colors)

    # Flatten and predict
    model = joblib.load(MODEL_PATH)
    flat_colors = colors.flatten().reshape(1, -1)
    mood = model.predict(flat_colors)[0]
    st.subheader(f"🧠 Predicted Mood: {mood}")
