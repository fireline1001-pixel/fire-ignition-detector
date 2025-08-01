import os
import tempfile

import cv2
import streamlit as st
import torch

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

# 기본 가중치 경로
DEFAULT_WEIGHT_PATH = "runs/train/ignition_yolo_final_retrain2/weights/best.pt"

# 이미지 표시 크기
IMG_DISPLAY_WIDTH = 800

# Streamlit 상단 디자인
st.set_page_config(layout="centered")
st.markdown("<h1 style='text-align: center;'>🔥 Ignition Point Detector</h1>", unsafe_allow_html=True)

# 로고 이미지 표시
if os.path.exists("logoall.jpg"):
    st.image("logoall.jpg", width=500)
else:
    st.warning("로고 이미지 (logoall.jpg)를 찾을 수 없습니다.")

# 가중치 파일 로드
weights = DEFAULT_WEIGHT_PATH
if not os.path.exists(weights):
    weights = st.file_uploader("YOLOv5 모델 가중치 (.pt)", type=["pt"])
    if weights:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp.write(weights.read())
            weights = tmp.name
    else:
        st.error("가중치 파일을 업로드해주세요.")
        st.stop()

# YOLOv5 모델 초기화
device = select_device("")
model = DetectMultiBackend(weights, device=device)
stride, names = model.stride, model.names
imgsz = check_img_size(640, s=stride)

# 🔍 이미지 업로드
uploaded_files = st.file_uploader(
    "이미지를 선택하세요 (다중 선택 가능)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"📷 {uploaded_file.name}")
        # 임시 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            tmp_img.write(uploaded_file.read())
            img_path = tmp_img.name

        # 이미지 처리
        dataset = LoadImages(img_path, img_size=imgsz, stride=stride)
        for path, im, im0s, _ in dataset:
            im = torch.from_numpy(im).to(device)
            im = im.float() / 255.0
            if im.ndimension() == 3:
                im = im.unsqueeze(0)

            pred = model(im)
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

            for i, det in enumerate(pred):
                im0 = im0s.copy()
                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = f"{names[int(cls)]} {conf:.2f}"
                        cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                        cv2.putText(
                            im0, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                        )

                # 결과 표시
                im0_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                st.image(im0_rgb, caption="예측 결과", width=IMG_DISPLAY_WIDTH)
