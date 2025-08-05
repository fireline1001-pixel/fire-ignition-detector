import streamlit as st
import tempfile
import os
from PIL import Image
import torch
import numpy as np
import cv2
import yaml
import sys

# yolov5 디렉토리를 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

st.set_page_config(page_title="Fire Detection using YOLOv5", layout="wide")
st.image("logoall.jpg", use_column_width=True)

st.title("🔥 Fire Detection using YOLOv5")
st.markdown("Upload a YOLOv5 model (.pt) and fire scene images to analyze.")

@st.cache_resource
def load_model(model_path):
    model = attempt_load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

@st.cache_resource
def load_yaml(path='yolov5/data/coco128.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

model_file = st.file_uploader("Upload YOLOv5 Model (.pt)", type=["pt"])
image_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if model_file and image_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
        tmp_model.write(model_file.read())
        tmp_model_path = tmp_model.name

    model = load_model(tmp_model_path)
    names = model.module.names if hasattr(model, 'module') else model.names

    for img_file in image_files:
        img = Image.open(img_file).convert("RGB")
        img_array = np.array(img)
        img0 = img_array.copy()

        img_resized = letterbox(img_array, new_shape=640)[0]
        img_resized = img_resized.transpose((2, 0, 1))[::-1]
        img_resized = np.ascontiguousarray(img_resized)
        img_tensor = torch.from_numpy(img_resized).to(torch.device('cpu'))
        img_tensor = img_tensor.float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        pred = non_max_suppression(model(img_tensor)[0], 0.25, 0.45, None, False)[0]

        for *xyxy, conf, cls in pred:
            label = f'{names[int(cls)]} {conf:.2f}'
            xyxy = [int(x.item()) for x in xyxy]
            cv2.rectangle(img0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
            cv2.putText(img0, label, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        st.image(img0, caption=img_file.name, use_column_width=True)
