import streamlit as st
import sys, os
from PIL import Image

# ✅ Add parent folder (task3) to Python path so we can import src/
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.infer import predict

st.title('Task 3 — Multimodal House Price (Demo)')

uploaded = st.file_uploader('Upload house image', type=['jpg', 'png'])
size = st.number_input('Size (sqft)', value=1200)
beds = st.number_input('Bedrooms', min_value=0, value=3)

if st.button('Predict'):
    if uploaded is None:
        st.error('Upload an image first.')
    else:
        # ✅ Save uploaded image into your dataset folder
        img_dir = r"D:\DevelopersHubCorporation\task3\data\task3_images"
        os.makedirs(img_dir, exist_ok=True)
        img_path = os.path.join(img_dir, "temp_task3.jpg")  # file path inside the folder

        with open(img_path, 'wb') as f:
            f.write(uploaded.getbuffer())

        # ✅ Check if model exists
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "task3_fusion.pth")
        if not os.path.exists(model_path):
            st.error("Model file not found! Please run `python src/train.py` first to train and save the model.")
        else:
            pred = predict(img_path, [size, beds])
            st.metric('Predicted price', f"${pred:,.0f}")
