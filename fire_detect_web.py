import streamlit as st
import torch
import tempfile
import shutil
import os
from PIL import Image, UnidentifiedImageError
import cv2
import numpy as np
from pathlib import Path
from models.yolo import DetectionModel
from torch.serialization import add_safe_globals

# 안전하게 YOLO 모델 허용
add_safe_globals({'models.yolo.DetectionModel': DetectionModel})

st.set_page_config(layout="wide")
st.title("🔥 Ignition Point Detector v2.0")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
model_url = st.text_input("Enter model file URL (.pt):", "")

def download_and_load_model(url):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp_path = tmp.name
        with st.spinner("Downloading model..."):
            with open(tmp_path, "wb") as f:
                f.write(Path(url).read_bytes() if Path(url).exists() else requests.get(url).content)
        model = torch.load(tmp_path, map_location=torch.device("cpu"), weights_only=False)
        return model

def infer_and_visualize(model, image):
    img = np.array(image)
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        preds = model(img_tensor)[0]
    boxes = preds[:, :4].numpy().astype(int)
    confs = preds[:, 4].numpy()
    for i, box in enumerate(boxes):
        cv2.rectangle(img_resized, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(img_resized, f"{confs[i]:.2f}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_resized

if uploaded_file and model_url:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=True)
        model = download_and_load_model(model_url)
        result = infer_and_visualize(model, image)
        st.image(result, caption="Detection Result", use_column_width=True)
    except UnidentifiedImageError:
        st.error("Invalid image file.")
    except Exception as e:
        st.error(f"Error: {e}")