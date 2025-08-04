import streamlit as st
import torch
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# 로고 및 제목
st.set_page_config(page_title="Ignition Point Detector", layout="wide")
st.image("logoall.jpg", use_column_width=True)

# 모델 로드 함수
@st.cache_resource
def load_model_from_uploaded_file(uploaded_model):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
        tmp.write(uploaded_model.getvalue())
        tmp_path = tmp.name
    return torch.load(tmp_path, map_location=torch.device('cpu'), weights_only=False)

# 이미지 예측 함수
def predict_image(model, image):
    temp_img_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
    image.save(temp_img_path)

    results = model([temp_img_path])
    results.render()

    img_array = results.ims[0]
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return img_array

# 모델 업로드
uploaded_model = st.file_uploader("📦 YOLOv5 모델 (.pt) 파일 업로드", type=['pt'])

if uploaded_model:
    try:
        model = load_model_from_uploaded_file(uploaded_model)
        st.success("모델이 성공적으로 로드되었습니다.")
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        st.stop()

    # 이미지 업로드
    uploaded_images = st.file_uploader("🖼 분석할 이미지 선택 (여러 개 가능)", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

    if uploaded_images:
        for uploaded_image in uploaded_images:
            st.markdown("---")
            image = Image.open(uploaded_image).convert('RGB')
            st.image(image, caption="원본 이미지", use_column_width=True)

            with st.spinner("🔍 예측 중..."):
                result_img = predict_image(model, image)
                st.image(result_img, caption="🔥 발화지점 예측 결과", use_column_width=True)
