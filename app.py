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
        binary_url = "https://drive.google.com/uc?id=19FgTMja2-xP-v3ljOtkoxWx-v6s1iTbs"
        gdown.download(binary_url, "dr_binary.h5", quiet=False)

    if not os.path.exists("dr_stage.h5"):
        stage_url = "https://drive.google.com/uc?id=16kvHprG6xxaHuuKQdW6QcYIZ0C2nuFwn"
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

with st.spinner("🔄 Loading models..."):
    binary_model, stage_model = load_models()

# --------------------------
# Step 3: UI
# --------------------------
st.set_page_config(page_title="Diabetic Retinopathy Predictor", layout="centered")
st.title("👁️ Diabetic Retinopathy Detection")
st.markdown("Upload a **retinal fundus image**, and this AI will detect DR and classify its stage (if any).")

uploaded_file = st.file_uploader("📤 Upload Fundus Image", type=["jpg", "jpeg", "png"])

# Stage-based suggestions
suggestions = {
    'Mild': [
        "🟢 Maintain good blood sugar control.",
        "🟢 Yearly eye exams recommended.",
        "🟢 Eat a balanced diet and exercise regularly."
    ],
    'Moderate': [
        "🟡 Eye checkups every 6 months.",
        "🟡 Control blood pressure and sugar strictly.",
        "🟡 Consider discussing laser treatment options."
    ],
    'Severe': [
        "🔴 Urgent appointment with eye specialist required.",
        "🔴 Possible need for laser or anti-VEGF therapy.",
        "🔴 Avoid smoking and monitor your vision closely."
    ],
    'Proliferative': [
        "⚠️ Immediate treatment needed to prevent blindness.",
        "⚠️ Laser surgery or vitrectomy may be required.",
        "⚠️ Frequent monitoring and follow-ups are critical."
    ]
}

# --------------------------
# Step 4: Prediction
# --------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Step 1: Binary Prediction
    binary_pred = binary_model.predict(img_array, verbose=0)[0][0]
    is_dr = binary_pred > 0.5

    st.subheader("🔍 Prediction Result")
    if is_dr:
        st.success("🩺 **Diabetic Retinopathy Detected!**")

        # Step 2: Stage Prediction
        stage_pred = stage_model.predict(img_array, verbose=0)
        stage_label = ['Mild', 'Moderate', 'Severe', 'Proliferative'][np.argmax(stage_pred)]

        st.markdown(f"🧪 **Stage Detected:** `{stage_label}`")
        st.bar_chart(stage_pred[0])

        st.subheader("📋 Precautions & Suggestions")
        for tip in suggestions[stage_label]:
            st.markdown(f"- {tip}")
    else:
        st.info("✅ **No signs of Diabetic Retinopathy (Healthy Retina)**")
        st.markdown("🔵 You can continue regular yearly eye checkups and maintain a healthy lifestyle.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + TensorFlow + EfficientNetB3")
