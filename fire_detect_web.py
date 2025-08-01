import streamlit as st
import torch
import os
from PIL import Image
import numpy as np
import tempfile
from pathlib import Path
import shutil
import cv2

# 🔧 Streamlit 설정
st.set_page_config(page_title="Ignition Point Detector", layout="centered")

# 🔺 상단 로고 이미지
st.image("logoall.jpg", use_column_width=True)

st.markdown("<h1 style='text-align: center;'>🔥 Ignition Point Detector</h1>", unsafe_allow_html=True)

# 🔺 YOLOv5 모델 업로드
st.subheader("YOLOv5 모델 가중치 (.pt)")
model_file = st.file_uploader("Drag and drop file here", type=["pt"], help=".pt 파일 업로드", label_visibility="collapsed")

# 🔺 이미지 업로드
st.subheader("분석할 화재 이미지 업로드")
uploaded_images = st.file_uploader("이미지 파일을 업로드하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# 내부 경로 설정용 임시 디렉토리
temp_dir = tempfile.mkdtemp()

# 🧠 모델 로딩
model = None
if model_file is not None:
    try:
        model_path = os.path.join(temp_dir, model_file.name)
        with open(model_path, "wb") as f:
            f.write(model_file.getbuffer())
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        st.success("✅ 모델이 성공적으로 로드되었습니다.")
    except Exception as e:
        st.error(f"❌ 모델 로딩 중 오류 발생: {e}")

# ▶️ 예측 실행
if model is not None and uploaded_images:
    st.subheader("🔍 예측 결과")

    for img_file in uploaded_images:
        try:
            image = Image.open(img_file).convert("RGB")
            img_np = np.array(image)
            results = model(img_np)

            # 결과 이미지 저장
            pred_img = np.squeeze(results.render())  # render() returns list with one image
            pred_pil = Image.fromarray(pred_img)

            st.image(pred_pil, caption=f"결과: {img_file.name}", use_column_width=True)

            # 이미지 다운로드
            download_path = os.path.join(temp_dir, f"result_{img_file.name}")
            pred_pil.save(download_path)
            with open(download_path, "rb") as f:
                st.download_button(
                    label="결과 이미지 다운로드",
                    data=f,
                    file_name=f"result_{img_file.name}",
                    mime="image/jpeg"
                )
        except Exception as e:
            st.error(f"{img_file.name} 처리 중 오류 발생: {e}")
else:
    if model is None:
        st.warning("YOLOv5 가중치 파일을 업로드해 주세요.")
    elif not uploaded_images:
        st.info("분석할 이미지를 업로드해 주세요.")

# 종료 시 임시 폴더 정리
import atexit
atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
