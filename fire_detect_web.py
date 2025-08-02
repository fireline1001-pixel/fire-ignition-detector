import streamlit as st
import torch
import tempfile
import os
import shutil
import cv2
from PIL import Image
import numpy as np

st.set_page_config(page_title="Ignition Point Detector", layout="centered")

# 상단 제목 및 로고
st.image("logoall.jpg", use_column_width=True)
st.markdown("<h1 style='text-align: center;'>🔥 발화점 검출기</h1>", unsafe_allow_html=True)

# 업로드 섹션
st.markdown("### YOLOv5 모델 가중치 (.pt)")
model_file = st.file_uploader("여기에 파일을 끌어다 놓습니다.", type=["pt"], key="model")

st.markdown("### 분석할 화재 이미지 업로드")
image_files = st.file_uploader("이미지 파일을 업로드하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="images")

# 예측 수행 함수
def load_model(pt_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=pt_path, force_reload=True)
    return model

def run_inference(model, image):
    results = model(image)
    return results

def draw_results(results, img_np):
    for *box, conf, cls in results.xyxy[0].tolist():
        x1, y1, x2, y2 = map(int, box)
        label = f'{results.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return img_np

# 예측 실행
if model_file and image_files:
    with tempfile.TemporaryDirectory() as tmpdir:
        # 모델 저장
        model_path = os.path.join(tmpdir, "model.pt")
        with open(model_path, "wb") as f:
            f.write(model_file.read())

        st.success("✅ 모델 로드 중입니다...")
        try:
            model = load_model(model_path)
            st.success("✅ 모델 로드 완료!")

            for uploaded_image in image_files:
                st.markdown("---")
                st.subheader(f"📷 입력 이미지: {uploaded_image.name}")
                image = Image.open(uploaded_image).convert("RGB")
                st.image(image, caption="업로드된 이미지", use_column_width=True)

                # PIL → OpenCV 변환
                img_np = np.array(image)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # 예측
                results = run_inference(model, img_bgr)

                # 결과 시각화
                img_result = draw_results(results, img_bgr.copy())
                img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)

                st.image(img_result_rgb, caption="🔍 예측 결과", use_column_width=True)
        except Exception as e:
            st.error(f"❌ 예측 중 오류 발생: {e}")
else:
    st.warning("YOLOv5 가중치 파일과 분석할 이미지를 업로드해주세요.")
