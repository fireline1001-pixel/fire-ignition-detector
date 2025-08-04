import streamlit as st
import torch
import tempfile
import shutil
import os
import cv2
from PIL import Image
import numpy as np

# 웹앱 제목과 설명
st.set_page_config(page_title="Ignition Point Detector", layout="centered")

st.markdown("""
    <div style='text-align: center;'>
        <img src='https://raw.githubusercontent.com/fireline1001-pixel/fire-ignition-detector/main/logoall.jpg' width='500'/>
        <h1>🔥 발화점 검출기</h1>
    </div>
""", unsafe_allow_html=True)

# 가중치 파일 업로드
st.subheader("YOLOv5 모델 가중치 (.pt)")
model_file = st.file_uploader("여기에 파일을 끌어다 놓습니다.", type=["pt"], key="model")

# 이미지 파일 업로드
st.subheader("분석할 화재 이미지 업로드")
image_files = st.file_uploader("이미지 파일을 업로드하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="image")

# 예측 버튼
if st.button("예측 실행"):
    if not model_file:
        st.warning("YOLOv5 가중치 파일을 업로드해 주세요.")
    elif not image_files:
        st.warning("분석할 이미지를 업로드해 주세요.")
    else:
        # 가중치 임시 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
            tmp_model.write(model_file.read())
            model_path = tmp_model.name

        # 모델 로드
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

        # 이미지 처리
        for img_file in image_files:
            # 임시 이미지 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                tmp_img.write(img_file.read())
                tmp_img_path = tmp_img.name

            # 이미지 열기 및 예측
            img = cv2.imread(tmp_img_path)
            results = model(img)

            # 결과 좌표 추출 및 시각화
            boxes = results.xyxy[0].cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(img, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 이미지 RGB로 변환 후 출력
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption=f"Prediction - {img_file.name}", use_container_width=True)

        # 임시 파일 삭제
        os.unlink(model_path)
        for img_file in image_files:
            try:
                os.unlink(img_file.name)
            except:
                pass
