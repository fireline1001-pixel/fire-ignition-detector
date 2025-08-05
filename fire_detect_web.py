import streamlit as st
import torch
import tempfile
import os
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from models.experimental import attempt_load

# 🔧 이미지 경로 가져오기 (logoall.jpg 포함)
def resource_path(relative_path):
    return os.path.join(os.path.dirname(__file__), relative_path)

# ✅ Streamlit 웹앱 제목 및 로고 표시
st.set_page_config(page_title="Fire Detection using YOLOv5", layout="centered")
st.markdown("<h1 style='text-align: center;'>🔥 Fire Detection using YOLOv5</h1>", unsafe_allow_html=True)

# ✅ 로고 이미지 안전하게 표시
logo_path = resource_path("logoall.jpg")
try:
    logo_image = Image.open(logo_path)
    st.image(logo_image, use_column_width=True)
except UnidentifiedImageError:
    st.warning("⚠️ 'logoall.jpg' 파일이 손상되었거나 잘못된 이미지입니다.")
except FileNotFoundError:
    st.warning("❌ 'logoall.jpg' 파일을 찾을 수 없습니다. 프로젝트 루트에 포함시켜주세요.")

# ✅ 파일 업로드 섹션
st.markdown("Upload a YOLOv5 model (.pt) and fire scene images to analyze.")

model_file = st.file_uploader("Upload YOLOv5 Model (.pt)", type=["pt"])
image_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# ✅ 모델 로드 함수
@st.cache_resource
def load_model(model_path):
    return attempt_load(model_path, map_location=torch.device('cpu'))

# ✅ 추론 및 시각화
def run_inference(model, image):
    img = np.array(image.convert("RGB"))
    img_resized = cv2.resize(img, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in det:
                label = f'Fire {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=(255, 0, 0), line_thickness=2)

    return img

# ✅ 실행 버튼
if model_file and image_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_file.read())
        model = load_model(tmp.name)

    for img_file in image_files:
        try:
            image = Image.open(img_file)
        except UnidentifiedImageError:
            st.error(f"❌ 이미지 파일을 열 수 없습니다: {img_file.name}")
            continue

        st.markdown(f"### 🔍 Analyzing: {img_file.name}")
        result_img = run_inference(model, image)
        st.image(result_img, caption="Detection Result", use_column_width=True)
