# 🌿 PlantPulse AI

**PlantPulse AI — an AI-powered monitoring system for early detection and smart treatment of plant diseases using intelligent technology.**

## 🚀 Overview

It"s a deep learning-based application that detects plant leaf diseases from images and provides basic treatment suggestions using a CNN model.

## 🧠 Model Training

* Dataset: PlantVillage (TensorFlow Datasets)
* Preprocessing: Image resizing (224x224), normalization
* Data Augmentation: Flip, rotation, zoom
* Model: MobileNetV2 (Transfer Learning) + custom layers
* Training: Adam optimizer, sparse categorical crossentropy, fine-tuning
* Model saved in `.keras` format for compatibility

## 💻 Application

* Built using Streamlit
* Two-page UI (Landing + Detection)
* Upload image → Predict disease → Display solution

## 🛠️ Technologies Used

Python, TensorFlow/Keras, Streamlit, NumPy, Pillow

## 📂 Project Structure

app.py
model.keras
requirements.txt
.python-version

## ▶️ Run Locally

pip install -r requirements.txt
streamlit run app.py

## 🌱 Future Work

Add camera input, improve accuracy, expand solution database, deploy mobile version

## 💚 Conclusion

Helps in early detection of plant diseases and supports smart, sustainable farming.
