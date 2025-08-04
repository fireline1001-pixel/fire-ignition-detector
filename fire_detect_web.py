import streamlit as st
import torch
import tempfile
import os
import shutil
from PIL import Image
import cv2
import numpy as np
from datetime import datetime

# 상단 로고 및 제목 표시
st.set_page_config(page_title="Ignition Point Detector", layout="centered")
st.markdown(
    """
    <div style='text-align: center; padding: 10px 0;'>
        <img src="https://raw.githubusercontent.com/fireline1001-pixel/fire-ignition-detector/main/logoall.jpg" width="300"/>
        <h2>Ignition Point Detector 🔥</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# 모델 업로드
st.sidebar.header("1️⃣ 모델 가중치 업로드")
uploaded_model = st.sidebar.file_uploader("YOLOv5 .pt 파일을 업로드하세요", type=["pt"])

# 이미지 업로드
st.sidebar.header("2️⃣ 이미지 업로드")
uploaded_images = st.sidebar.file_uploader("분석할 이미지 파일 업로드 (다중 선택 가능)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# 모델 로딩
@st.cache_resource
def load_model_from_uploaded_file(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "model.pt")
    with open(model_path, "wb") as f:
        f.write(uploaded_file.read())
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

# 예측 수행
def run_inference(model, image_pil):
    img = np.array(image_pil)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model([img_rgb], size=640)
    return results

# 이미지에 바운딩 박스 그리기
def draw_boxes(image_pil, results):
    img = np.array(image_pil).copy()
    for *xyxy, conf, cls in results.xyxy[0].tolist():
        label = f"{results.names[int(cls)]} {conf:.2f}"
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return Image.fromarray(img)

# 메인 실행
if uploaded_model and uploaded_images:
    model = load_model_from_uploaded_file(uploaded_model)

    st.header("3️⃣ 예측 결과")

    for uploaded_image in uploaded_images:
        st.subheader(f"🔍 분석 중: {uploaded_image.name}")
        image_pil = Image.open(uploaded_image).convert("RGB")
        results = run_inference(model, image_pil)
        image_with_boxes = draw_boxes(image_pil, results)

        st.image(image_with_boxes, caption=f"📍 예측 결과 - {uploaded_image.name}", use_column_width=True)

        # 다운로드 버튼
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        temp_img_path = f"result_{timestamp}.jpg"
        image_with_boxes.save(temp_img_path)
        with open(temp_img_path, "rb") as f:
            btn = st.download_button(
                label="📥 결과 이미지 다운로드",
                data=f,
                file_name=f"predicted_{uploaded_image.name}",
                mime="image/jpeg"
            )
        os.remove(temp_img_path)

elif not uploaded_model:
    st.info("왼쪽 사이드바에서 YOLOv5 가중치(.pt) 파일을 먼저 업로드하세요.")
elif not uploaded_images:
    st.info("왼쪽 사이드바에서 이미지 파일을 업로드하세요.")
