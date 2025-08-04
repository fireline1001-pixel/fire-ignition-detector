import streamlit as st
import torch
import tempfile
import os
import cv2
import numpy as np
from PIL import Image

# 제목 및 로고
st.set_page_config(page_title="Ignition Point Detector", layout="centered")
st.image("logoall.jpg", use_container_width=True)
st.markdown("## 🔥 발화점 검출기")

# 모델 파일 업로드
st.subheader("YOLOv5 모델 가중치 (.pt)")
uploaded_model = st.file_uploader("여기에 파일을 끌어다 놓습니다.", type=["pt"], key="model")

# 이미지 파일 업로드
st.subheader("분석할 화재 이미지 업로드")
uploaded_images = st.file_uploader(
    "이미지 파일을 업로드하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="images"
)

# 모델 로드 함수
@st.cache_resource
def load_model_from_uploaded_file(uploaded_model_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model_file:
        temp_model_file.write(uploaded_model_file.read())
        temp_model_path = temp_model_file.name

    # weights_only=False 를 명시하여 전체 모델 로드 허용 (신뢰할 수 있는 파일에 한함)
    model = torch.load(temp_model_path, map_location=torch.device("cpu"), weights_only=False)
    model.eval()
    return model

# 예측 함수
def detect_and_display(model, image_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img_file:
        temp_img_file.write(image_file.read())
        temp_img_path = temp_img_file.name

    img = cv2.imread(temp_img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img_rgb)
    results.render()

    for im in results.ims:
        st.image(im, caption="📌 예측 결과", use_column_width=True)

# 실행
if uploaded_model and uploaded_images:
    try:
        model = load_model_from_uploaded_file(uploaded_model)
        for uploaded_image in uploaded_images:
            st.markdown(f"**파일명:** {uploaded_image.name}")
            detect_and_display(model, uploaded_image)
    except Exception as e:
        st.error(f"모델 로딩 또는 예측 중 오류 발생: {e}")
else:
    st.warning("YOLOv5 가중치 파일과 분석할 이미지를 업로드해주세요.")
