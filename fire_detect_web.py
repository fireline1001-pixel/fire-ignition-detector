import streamlit as st
import torch
from PIL import Image
import tempfile
import os
import shutil
import uuid
from datetime import datetime
import cv2
import numpy as np

# 기본 설정
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>🔥 발화점 검출기</h1>", unsafe_allow_html=True)
st.image("logoall.jpg", use_container_width=True)

# 업로드 박스
st.subheader("📦 YOLOv5 모델 (.pt) 파일 업로드")
uploaded_model = st.file_uploader("Drag and drop file here", type=["pt"])

st.subheader("🖼️ 분석할 화재 이미지 업로드")
uploaded_images = st.file_uploader("Drag and drop files here", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# 모델 로드 함수
@st.cache_resource
def load_model_from_uploaded_file(model_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
        tmp_file.write(model_file.read())
        tmp_file_path = tmp_file.name

    try:
        # PyTorch 2.6 대응: weights_only=False 명시
        model = torch.load(tmp_file_path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None

# 예측 함수
def run_detection(model, image_file):
    image = Image.open(image_file).convert("RGB")
    image_np = np.array(image)

    results = model(image_np)
    results.render()

    # 결과 이미지 저장
    result_image = Image.fromarray(results.ims[0])
    return result_image

# 실행 버튼
if uploaded_model and uploaded_images:
    model = load_model_from_uploaded_file(uploaded_model)
    if model:
        st.success("모델이 성공적으로 로드되었습니다. 예측을 시작합니다.")

        for img_file in uploaded_images:
            st.markdown("---")
            st.markdown(f"#### 🔍 분석 중: `{img_file.name}`")
            pred_img = run_detection(model, img_file)
            st.image(pred_img, caption=f"예측 결과 - {img_file.name}", use_container_width=True)
else:
    st.warning("YOLOv5 가중치 파일과 분석할 이미지를 업로드해주세요.")
