import streamlit as st
import torch
import tempfile
import shutil
import os
import cv2
from PIL import Image
import numpy as np

# 페이지 설정
st.set_page_config(page_title="Ignition Point Detector", page_icon="🔥", layout="wide")

# 상단 로고 및 제목 표시
st.image("logoall.jpg", use_container_width=True)

st.markdown(
    "<h1 style='text-align: center;'>🔥 발화점 검출기</h1>", unsafe_allow_html=True
)

# YOLOv5 모델 업로드
st.markdown("### YOLOv5 모델 가중치 (.pt)")
model_file = st.file_uploader("여기에 파일을 끌어다 놓습니다.", type=["pt"])

# 분석할 이미지 업로드
st.markdown("### 분석할 화재 이미지 업로드")
image_files = st.file_uploader("이미지 파일을 업로드하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# 임시 디렉토리 생성
if model_file is not None and image_files:
    with tempfile.TemporaryDirectory() as tmpdir:
        # 모델 저장
        model_path = os.path.join(tmpdir, model_file.name)
        with open(model_path, "wb") as f:
            f.write(model_file.read())

        # 모델 로드
        try:
            model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
        except Exception as e:
            st.error(f"모델 로딩 실패: {e}")
            st.stop()

        # 이미지 저장 및 예측
        st.markdown("### 🔍 예측 결과")

        for uploaded_file in image_files:
            # 파일 저장
            img_path = os.path.join(tmpdir, uploaded_file.name)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.read())

            # 이미지 로드 및 예측
            results = model(img_path)
            results.render()  # 예측 박스를 그린 이미지 생성

            # 결과 시각화
            for im in results.ims:
                im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                st.image(im_rgb, caption="예측 결과", use_container_width=True)

elif model_file is None or not image_files:
    st.warning("YOLOv5 가중치 파일과 분석할 이미지를 업로드해주세요.")
