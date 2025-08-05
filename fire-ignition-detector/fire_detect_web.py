
import streamlit as st
import torch
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
from utils.general import non_max_suppression, scale_coords
from models.experimental import attempt_load

st.set_page_config(layout="centered")
st.image("logoall.jpg", use_column_width=True)

st.title("🔥 Fire Detection using YOLOv5")
st.markdown("Upload a YOLOv5 model (.pt) and fire scene images to analyze.")

@st.cache_resource
def load_model(path):
    model = attempt_load(path, map_location=torch.device("cpu"))
    return model

uploaded_model = st.file_uploader("Upload YOLOv5 Model (.pt)", type=["pt"])
uploaded_images = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_model is not None and uploaded_images:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model_file:
        tmp_model_file.write(uploaded_model.read())
        tmp_model_path = tmp_model_file.name

    model = load_model(tmp_model_path)

    for uploaded_image in uploaded_images:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

        pred = model(img_tensor, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img.shape).round()

            for *xyxy, conf, cls in pred:
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=uploaded_image.name, use_column_width=True)
