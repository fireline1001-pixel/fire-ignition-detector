
import streamlit as st
import torch
import tempfile
import os
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Fire Detection Web App", layout="wide")

st.image("logoall.jpg", use_column_width=True)

st.title("🔥 Fire Detection using YOLOv5")
st.markdown("Upload a YOLOv5 model (.pt) and fire scene images to analyze.")

@st.cache_resource
def load_model(model_path):
    return torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)

model_file = st.file_uploader("Upload YOLOv5 Model (.pt)", type=["pt"])
image_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if model_file and image_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
        tmp_model.write(model_file.read())
        tmp_model_path = tmp_model.name

    model = load_model(tmp_model_path)
    model.eval()

    for img_file in image_files:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = model(img)
        rendered_img = results.render()[0]
        st.image(rendered_img, channels="BGR", caption=img_file.name, use_container_width=True)
