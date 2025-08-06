import streamlit as st

st.title("Ignition Point Detector")

pt_url = st.text_input("Enter your .pt file URL (direct download link):")

if pt_url:
    st.success(f"Model URL received: {pt_url}")
    st.info("⚠️ 모델 다운로드 및 추론 코드는 여기에 추가되어야 합니다.")
else:
    st.warning("모델(.pt) 파일의 직접 다운로드 URL을 입력해주세요.")
