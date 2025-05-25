import streamlit as st
import numpy as np
import cv2
import gdown
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# --------------------------
# Step 1: Download models
# --------------------------
def download_models():
    if not os.path.exists("dr_binary.h5"):
        binary_url = "https://drive.google.com/uc?id=YOUR_BINARY_MODEL_ID"
        gdown.download(binary_url, "dr_binary.h5", quiet=False)

    if not os.path.exists("dr_stage.h5"):
        stage_url = "https://drive.google.com/uc?id=YOUR_STAGE_MODEL_ID"
        gdown.download(stage_url, "dr_stage.h5", quiet=False)

# --------------------------
# Step 2: Load models
# --------------------------
@st.cache_resource
def load_models():
    download_models()
    binary_model = load_model("dr_binary.h5")
    stage_model = load_model("dr_stage.h5")
    return binary_model, stage_model

binary_model, stage_model = load_models()

# --------------------------
# Step 3: UI
# --------------------------
st.set_page_config(page_title="Diabetic Retinopathy Predictor", layout="centered")
st.title("👁️ Diabetic Retinopathy Detection")
st.markdown("Upload a **fundus image**, and this AI will detect DR and classify its stage if applicable.")

uploaded_file = st.file_uploader("Upload a Fundus Image", type=["jpg", "jpeg", "png"])

# --------------------------
# Step 4: Prediction
# --------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Step 1: Binary Prediction
    binary_pred = binary_model.predict(img_array, verbose=0)[0][0]
    is_dr = binary_pred > 0.5

    if is_dr:
        st.subheader("🩺 Prediction: **Diabetic Retinopathy Detected**")

        # Step 2: Stage Prediction
        stage_pred = stage_model.predict(img_array, verbose=0)
        stage_label = ['Mild', 'Moderate', 'Severe', 'Proliferative'][np.argmax(stage_pred)]

        st.markdown(f"🧪 **Stage:** `{stage_label}`")
        st.bar_chart(stage_pred[0])
    else:
        st.subheader("✅ Prediction: **No Diabetic Retinopathy**")

