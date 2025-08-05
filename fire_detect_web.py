import streamlit as st
import torch
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="발화점 검출기", layout="centered")

# 상단 디자인
st.markdown(
    """
    <div style="text-align: center;">
        <h1>🔥 발화점 검출기</h1>
        <h3>Ignition Point Detector v1.6.0[인천소방]</h3>
        <img src="https://fire-ignition-detector.onrender.com/static/logoall.jpg" width="300">
    </div>
    """,
    unsafe_allow_html=True
)

# 모델 파일 업로드
st.subheader("📦 YOLOv5 모델 (.pt) 파일 업로드")
uploaded_model = st.file_uploader(
    "Drag and drop file here",
    type=["pt"],
    key="model",
    label_visibility="collapsed"
)

# 이미지 파일 업로드
st.subheader("🖼️ 분석할 화재 이미지 업로드")
uploaded_images = st.file_uploader(
    "Drag and drop files here",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="images",
    label_visibility="collapsed"
)

# 모델 로드 함수 (캐시 적용)
@st.cache_resource
def load_model_from_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model_file:
        temp_model_file.write(uploaded_file.read())
        temp_model_path = temp_model_file.name
    model = torch.load(temp_model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    return model

# 이미지 예측 함수
def predict_image(model, image):
    img_array = np.array(image)
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    results = model(img_array)
    return results

# 예측 실행
if uploaded_model and uploaded_images:
    try:
        model = load_model_from_uploaded_file(uploaded_model)

        for uploaded_image in uploaded_images:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="원본 이미지", use_container_width=True)

            # 예측
            results = predict_image(model, image)
            results.render()  # 이미지 위에 예측 박스 렌더링
            rendered_img = Image.fromarray(results.ims[0])

            st.image(rendered_img, caption="예측 결과", use_container_width=True)

    except Exception as e:
        st.error(f"오류 발생: {e}")
else:
    st.warning("YOLOv5 가중치 파일과 분석할 이미지를 업로드해주세요.")
