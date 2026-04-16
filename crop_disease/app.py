import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PlantPulse AI", page_icon="🌿")

# -------- LOAD MODEL --------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fixed_model.h5", compile=False)

model = load_model()

# -------- CLASS NAMES --------
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry___healthy',
    'Cherry___Powdery_mildew', 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust', 'Corn___healthy', 'Corn___Northern_Leaf_Blight',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy',
    'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___healthy',
    'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# -------- SOLUTIONS --------
solutions = {
    "Tomato___Early_blight": "Remove infected leaves and apply fungicide",
    "Potato___Late_blight": "Avoid overwatering and use fungicide",
    "Apple___Black_rot": "Prune affected areas and spray fungicide"
}

# -------- PAGE CONTROL --------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_detect():
    st.session_state.page = "detect"

def go_home():
    st.session_state.page = "home"

# -------- HOME PAGE --------
if st.session_state.page == "home":
    st.markdown("## 🌍 Go Green 🌿")
    st.write("### Healthy plants, healthy life")
    st.caption("Your smart assistant for plant health 🌱")
    st.button("🚀 Get Started", on_click=go_detect)

# -------- DETECTION PAGE --------
elif st.session_state.page == "detect":

    st.markdown("## 🍃 Check whether your plant is healthy or diseased")

    uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image)

            # Preprocess
            img = image.resize((224, 224))
            img = np.array(img).astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            if st.button("🔍 Predict"):
                prediction = model.predict(img)
                idx = int(np.argmax(prediction))
                result = class_names[idx]

                st.success(f"✅ Prediction: {result}")

                solution = solutions.get(
                    result,
                    "🌱 General care: Maintain plant hygiene and use organic fertilizers"
                )

                st.warning(f"💡 Solution: {solution}")

        except Exception as e:
            st.error("Error processing image")

    st.button("⬅ Back", on_click=go_home)