
import streamlit as st
import tempfile
import os
from PIL import Image
import torch
import numpy as np
import cv2

st.set_page_config(page_title="Fire Detection using YOLOv5", layout="wide")

st.image("logoall.jpg", use_container_width=True)

st.title("🔥 Fire Detection using YOLOv5")
st.markdown("Upload a YOLOv5 model (.pt) and fire scene images to analyze.")

@st.cache_resource
def load_model(model_path):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

model_file = st.file_uploader("Upload YOLOv5 Model (.pt)", type=["pt"])
image_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if model_file and image_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
        tmp_model.write(model_file.read())
        tmp_model_path = tmp_model.name

    model = load_model(tmp_model_path)

    for img_file in image_files:
        img = Image.open(img_file).convert("RGB")
        img_array = np.array(img)

        results = model(img_array)
        results.render()
        rendered_img = Image.fromarray(results.ims[0])

        st.image(rendered_img, caption=img_file.name, use_container_width=True)
