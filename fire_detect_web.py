import streamlit as st
import torch
import cv2
import tempfile
import os
import numpy as np
from PIL import Image
from datetime import datetime

st.set_page_config(layout="wide")

# 🔷 로고 표시
st.image("logoall.jpg", use_column_width=True)

st.markdown(
    "<h1 style='text-align: center;'>🔥 Ignition Point Detector</h1>",
    unsafe_allow_html=True,
)

# 🔷 모델 가중치 업로드
st.subheader("YOLOv5 모델 가중치 (.pt)")
model_file = st.file_uploader(
    "Drag and drop file here",
    type=["pt"],
    key="pt_upload",
    label_visibility="collapsed"
)

# 🔷 모델 로딩 함수 (캐싱 포함)
@st.cache_resource
def load_model_from_file(uploaded_pt_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
            tmp_file.write(uploaded_pt_file.read())
            tmp_model_path = tmp_file.name
        model = torch.hub.load("ultralytics/yolov5", "custom", path=tmp_model_path, force_reload=True)
        return model
    except Exception as e:
        st.error(f"모델 로딩 실패: {e}")
        return None

# 모델이 성공적으로 로드되었는지 확인
model = None
if model_file:
    with st.spinner("YOLOv5 모델 로딩 중..."):
        model = load_model_from_file(model_file)
    if model:
        st.success("✅ 모델이 성공적으로 로딩되었습니다!")

# 🔷 분석할 이미지 업로드
st.subheader("분석할 화재 이미지 업로드")
image_files = st.file_uploader(
    "이미지 파일을 업로드하세요",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# 🔷 예측 및 결과 표시
if model and image_files:
    for uploaded_image in image_files:
        image = Image.open(uploaded_image).convert("RGB")
        img_np = np.array(image)

        with st.spinner(f"🔍 {uploaded_image.name} 분석 중..."):
            results = model(img_np)

        # 🔸 예측 결과 이미지 가져오기
        pred_img = np.squeeze(results.render())

        st.image(pred_img, caption=f"📌 분석 결과 - {uploaded_image.name}", use_column_width=True)

        # 🔸 저장 버튼
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f"result_{os.path.splitext(uploaded_image.name)[0]}_{timestamp}.jpg"
        cv2.imwrite(save_filename, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
        with open(save_filename, "rb") as f:
            st.download_button(
                label="💾 결과 이미지 다운로드",
                data=f,
                file_name=save_filename,
                mime="image/jpeg"
            )
        os.remove(save_filename)
else:
    if not model:
        st.warning("YOLOv5 가중치 파일을 업로드해 주세요.")
    elif not image_files:
        st.info("분석할 이미지를 업로드해 주세요.")
