import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PlantPulse AI", page_icon="🌿")

# -------- LOAD MODEL --------
import os

@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "final_streamlit_model.h5")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# -------- CLASS NAMES --------
class_names = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust',
    'Apple___healthy','Blueberry___healthy','Cherry___healthy',
    'Cherry___Powdery_mildew','Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust','Corn___healthy','Corn___Northern_Leaf_Blight',
    'Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot',
    'Peach___healthy','Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy','Potato___Early_blight','Potato___healthy',
    'Potato___Late_blight','Raspberry___healthy','Soybean___healthy',
    'Squash___Powdery_mildew','Strawberry___healthy',
    'Strawberry___Leaf_scorch','Tomato___Bacterial_spot',
    'Tomato___Early_blight','Tomato___healthy','Tomato___Late_blight',
    'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot','Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# -------- SOLUTIONS --------
solutions = {
    "Tomato___Early_blight": "Remove infected leaves and apply fungicide.",
    "Potato___Late_blight": "Avoid overwatering and use fungicide.",
    "Apple___Black_rot": "Prune affected areas and spray fungicide.",
    "Strawberry___Leaf_scorch": "Remove infected leaves and avoid overhead watering."
}

# -------- UI --------
st.title("🌿 PlantPulse AI")
st.write("Upload a leaf image to detect plant disease and get solution")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    if st.button("Predict"):
        pred = model.predict(img)
        idx = int(np.argmax(pred))
        result = class_names[idx]

        # Show prediction
        st.success(f"✅ Prediction: {result}")

        # Get solution
        solution = solutions.get(
            result,
            "🌱 General care: Maintain plant hygiene, proper watering, and use organic fertilizers."
        )

        # Show solution
        st.warning(f"💡 Solution: {solution}")
        