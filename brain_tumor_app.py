import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ================================
# Load trained model
# ================================
MODEL_PATH = "Brain_CNN2.h5"  # update if needed
model = tf.keras.models.load_model(MODEL_PATH)

# Define parameters
IMG_SIZE = (224, 224)
CLASS_NAMES = ["No Tumor", "Tumor"]  # update if label order differs

# ================================
# Helper function: preprocess image
# ================================
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# Custom CSS for colorful design + reliable background
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/premium-photo/3d-brain-scan-neural-connections-ai-background_1035770-31.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.8);
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #FDFEFE;
    text-align: center;
    text-shadow: 2px 2px 4px #000000;
}
.subtitle {
    font-size: 18px;
    color: #ECF0F1;
    text-align: center;
    margin-bottom: 20px;
    text-shadow: 1px 1px 3px #000000;
}
.stButton button {
    background-color: #2ECC71;
    color: white;
    border-radius: 10px;
    font-size: 16px;
    padding: 10px 20px;
}
.stFileUploader label {
    font-size: 16px !important;
    color: #FDFEFE !important;
    font-weight: bold;
    text-shadow: 1px 1px 2px #000000;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("<div class='title'>🧠 Brain Tumor Classification (CT/MRI)</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a brain CT or MRI scan to classify as <b style='color:#FADBD8;'>Tumor</b> or <b style='color:#D5F5E3;'>No Tumor</b>.</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image with border
    image = Image.open(uploaded_file)
    st.markdown("### 📷 Uploaded Scan:")
    st.image(image, caption="Uploaded Scan", use_container_width=True)

    # Preprocess and predict
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)

    # If softmax with 2 classes
    if prediction.shape[1] == 2:
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
    else:
        # If single sigmoid output
        predicted_class = 1 if prediction[0][0] > 0.5 else 0
        confidence = float(prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0])

    # Show results with styled boxes
    st.markdown("### 🔍 Prediction Result:")
    if predicted_class == 1:
        st.markdown(f"""<div style='background-color:rgba(192, 57, 43, 0.8); padding:15px; border-radius:10px;'>
                        <h3 style='color:#FADBD8;'>⚠️ Tumor Detected!</h3>
                        <p><b>Confidence:</b> {confidence*100:.2f}%</p>
                        <p>Please consult a radiologist.</p>
                        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div style='background-color:rgba(39, 174, 96, 0.8); padding:15px; border-radius:10px;'>
                        <h3 style='color:#D5F5E3;'>✅ No Tumor Detected</h3>
                        <p><b>Confidence:</b> {confidence*100:.2f}%</p>

                        </div>""", unsafe_allow_html=True)
