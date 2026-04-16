import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PlantPulse AI", page_icon="🌿")

# -------- LOAD MODEL --------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_model.h5", compile=False)

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

# -------- UI --------
st.title("🌿 PlantPulse AI")
st.write("Upload a leaf image to detect plant disease")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    if st.button("Predict"):
        pred = model.predict(img)
        result = class_names[np.argmax(pred)]
        st.success(f"Prediction: {result}")