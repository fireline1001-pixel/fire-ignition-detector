
import streamlit as st
import torch
import urllib.request
import os
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="centered")
st.image("logoall.jpg", use_column_width=True)
st.title("🔥 Fire Detection using YOLOv5")
st.markdown("The model will be downloaded automatically from Google Drive.")

@st.cache_resource
def download_and_load_model():
    model_dir = "weights"
    model_path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        url = "https://drive.google.com/uc?id=1Rpvz9ojAp-fV-4MRxrs0Oc_JUO5SD2Bh&export=download"
        urllib.request.urlretrieve(url, model_path)
    return torch.load(model_path, map_location=torch.device("cpu"))

model = download_and_load_model()
model.eval()

uploaded_images = st.file_uploader("Upload fire scene images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_images:
    for uploaded_file in uploaded_images:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_resized = cv2.resize(image_np, (640, 640))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            pred = model(image_tensor)[0]

        image_draw = image_resized.copy()
        for p in pred:
            if p is not None and len(p) >= 6:
                x1, y1, x2, y2, conf, cls = map(int, p[:6])
                cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image_draw, f"Fire {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        st.image(cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB), caption=uploaded_file.name, use_column_width=True)
