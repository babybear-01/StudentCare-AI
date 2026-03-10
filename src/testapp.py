import os
import joblib
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "student_model_v2_3.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None # ยังไม่มีโมเดล
    return joblib.load(MODEL_PATH)

model = load_model()

if model is None:
    st.warning("⚠️ กำลังเตรียมระบบ: ยังไม่พบโมเดล กรุณาอัปโหลดไฟล์โมเดลไปที่โฟลเดอร์ models/")
else:
    # โค้ดส่วนที่ใช้ทำนายผล
    pass