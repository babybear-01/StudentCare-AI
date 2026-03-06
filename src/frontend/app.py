import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import io
import json

# ==========================================
# 1. ตั้งค่าหน้าเพจ
# ==========================================
st.set_page_config(
    page_title="EduCare AI | Student Success Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. แถบเมนูด้านข้าง (Sidebar) เลือกโหมด
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135810.png", width=100)
    st.title("EduCare AI")
    st.markdown("**ระบบวิเคราะห์ความเสี่ยงผู้เรียน**")
    st.divider()
    
    # 🌟 พระเอกของเรา: ตัวเลือก 2 โหมด
    app_mode = st.radio(
        "📌 เลือกโหมดการทำงาน:",
        ["👤 วิเคราะห์รายบุคคล (Individual)", "📁 ประเมินยกชั้นเรียน (Batch Upload)"]
    )
    
    st.divider()
    st.markdown("👨‍💻 **พัฒนาโดย:** [ชื่อทีม/ชื่อคุณ]")
    st.markdown("ภาควิชา IT สาขา AIT - KMITL")

# ==========================================
# โหมดที่ 1: วิเคราะห์รายบุคคล (Individual Mode)
# ==========================================
if app_mode == "👤 วิเคราะห์รายบุคคล (Individual)":
    st.title("📊 ระบบประเมินความเสี่ยงรายบุคคล")
    st.markdown("กรอกข้อมูลนักเรียน 1 ราย เพื่อประเมินแนวโน้มผลการเรียน")

    with st.expander("📝 คลิกเพื่อกรอกข้อมูลนักเรียน", expanded=True):
        with st.form("prediction_form"):
            tab1, tab2, tab3 = st.tabs(["👤 ข้อมูลส่วนตัว & ครอบครัว", "📚 ข้อมูลการเรียน", "🏃 พฤติกรรม & สุขภาพ"])

            with tab1:
                col1, col2, col3 = st.columns(3)
                sex = col1.selectbox("เพศ (sex)", ["F", "M"])
                age = col2.slider("อายุ (age)", 15, 22, 16)
                address = col3.selectbox("ที่พักอาศัย (address)", ["U", "R"])
                famsize = col1.selectbox("ขนาดครอบครัว (famsize)", ["GT3", "LE3"])
                Pstatus = col2.selectbox("สถานะพ่อแม่ (Pstatus)", ["T", "A"])
                guardian = col3.selectbox("ผู้ปกครองหลัก (guardian)", ["mother", "father", "other"])
                Medu = col1.selectbox("การศึกษาแม่ (Medu: 0-4)", [0, 1, 2, 3, 4], index=4)
                Fedu = col2.selectbox("การศึกษาพ่อ (Fedu: 0-4)", [0, 1, 2, 3, 4], index=4)
                Mjob = col1.selectbox("อาชีพแม่ (Mjob)", ["teacher", "health", "services", "at_home", "other"])
                Fjob = col2.selectbox("อาชีพพ่อ (Fjob)", ["teacher", "health", "services", "at_home", "other"], index=1)

            with tab2:
                col4, col5, col6 = st.columns(3)
                school = col4.selectbox("โรงเรียน (school)", ["GP", "MS"])
                course = col5.selectbox("วิชาหลัก (course)", ["math", "portuguese"])
                reason = col6.selectbox("เหตุผลที่เลือกเรียน (reason)", ["course", "home", "reputation", "other"])
                traveltime = col4.selectbox("เวลาเดินทาง (traveltime: 1-4)", [1, 2, 3, 4])
                studytime = col5.selectbox("เวลาอ่านหนังสือ (studytime: 1-4)", [1, 2, 3, 4], index=1)
                failures = col6.number_input("ประวัติสอบตกวิชาอื่น (failures)", min_value=0, max_value=4, value=0)
                schoolsup = col4.selectbox("รับการสนับสนุนจากโรงเรียน (schoolsup)", ["yes", "no"], index=1)
                famsup = col5.selectbox("ครอบครัวสนับสนุน (famsup)", ["yes", "no"])
                paid = col6.selectbox("เรียนพิเศษ (paid)", ["yes", "no"], index=1)
                activities = col4.selectbox("ร่วมกิจกรรมเสริม (activities)", ["yes", "no"])
                nursery = col5.selectbox("เคยเรียนอนุบาล (nursery)", ["yes", "no"])
                higher = col6.selectbox("ตั้งใจเรียนต่อ ป.ตรี (higher)", ["yes", "no"])

            with tab3:
                col7, col8, col9 = st.columns(3)
                internet = col7.selectbox("มีอินเทอร์เน็ตที่บ้าน (internet)", ["yes", "no"])
                romantic = col8.selectbox("มีความสัมพันธ์ฉันชู้สาว (romantic)", ["yes", "no"], index=1)
                health = col9.slider("สุขภาพโดยรวม (health: 1-5)", 1, 5, 5)
                famrel = col7.slider("ความสัมพันธ์ในครอบครัว (famrel: 1-5)", 1, 5, 4)
                freetime = col8.slider("เวลาว่าง (freetime: 1-5)", 1, 5, 3)
                goout = col9.slider("ออกไปเที่ยวกับเพื่อน (goout: 1-5)", 1, 5, 2)
                Dalc = col7.slider("ดื่มแอลกอฮอล์วันธรรมดา (Dalc: 1-5)", 1, 5, 1)
                Walc = col8.slider("ดื่มแอลกอฮอล์วันหยุด (Walc: 1-5)", 1, 5, 1)
                absences = col9.number_input("จำนวนวันขาดเรียน (absences)", min_value=0, max_value=93, value=0)

            submit_button = st.form_submit_button(label="🔍 ประมวลผลด้วย AI", use_container_width=True)

    if submit_button:
        input_data = {
            "school": school, "sex": sex, "age": age, "address": address, "famsize": famsize,
            "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu, "Mjob": Mjob, "Fjob": Fjob,
            "reason": reason, "guardian": guardian, "traveltime": traveltime, "studytime": studytime,
            "failures": failures, "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
            "activities": activities, "nursery": nursery, "higher": higher, "internet": internet,
            "romantic": romantic, "famrel": famrel, "freetime": freetime, "goout": goout,
            "Dalc": Dalc, "Walc": Walc, "health": health, "absences": absences, "course": course
        }
        
        st.divider()
        st.header("📈 รายงานวิเคราะห์เชิงลึก (AI Insight Report)")
        with st.spinner("🧠 AI กำลังประมวลผล..."):
            try:
                response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
                if response.status_code == 200:
                    result = response.json()
                    is_high_risk = result["prediction_class"] == 1
                    
                    res_col1, res_col2 = st.columns([1, 1.5])
                    with res_col1:
                        st.subheader("ระดับความเสี่ยง")
                        gauge_val = 85 if is_high_risk else 15
                        gauge_color = "#FF4B4B" if is_high_risk else "#00CC96"
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number", value = gauge_val,
                            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': gauge_color}}
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if is_high_risk:
                            st.error("🚨 **สถานะ: เสี่ยงสูง (High Risk)**")
                        else:
                            st.success("✅ **สถานะ: ปกติ (Low Risk)**")

                    with res_col2:
                        st.subheader("📋 แนวทางดำเนินการ (Action Plan)")
                        if is_high_risk:
                            st.warning(f"⚠️ **ปัจจัยเฝ้าระวัง:** ขาดเรียน {absences} วัน | สอบตก {failures} ครั้ง")
                            st.error("1. นัดหมายนักเรียนพูดคุยด่วน\n2. ประสานงานผู้ปกครอง\n3. จัดสอนเสริม")
                        else:
                            st.success("1. รักษามาตรฐานการเรียน\n2. ส่งเสริมกิจกรรมพัฒนาศักยภาพ")
                else:
                    st.error("❌ เกิดข้อผิดพลาดจาก Backend API")
            except Exception as e:
                st.error("🔌 ไม่สามารถติดต่อเซิร์ฟเวอร์ได้ กรุณาตรวจสอบ FastAPI")

# ==========================================
# โหมดที่ 2: ประเมินยกชั้นเรียน (Batch Mode)
# ==========================================
elif app_mode == "📁 ประเมินยกชั้นเรียน (Batch Upload)":
    st.title("📁 ระบบประเมินความเสี่ยงยกชั้นเรียน (Batch Prediction)")
    st.markdown("อัปโหลดไฟล์ข้อมูลนักเรียน (CSV) เพื่อให้ AI ประเมินความเสี่ยงทั้งห้องในครั้งเดียว")
    
    # 1. กล่องอัปโหลดไฟล์
    uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์ CSV (รูปแบบเดียวกับ student-mat.csv)", type=["csv"])
    
    if uploaded_file is not None:
        # อ่านไฟล์ CSV
        df = pd.read_csv(uploaded_file)
        
        # ตรวจสอบเบื้องต้นว่ามีคอลัมน์สำคัญไหม
        if "absences" not in df.columns or "failures" not in df.columns:
            st.error("❌ ไฟล์ที่อัปโหลดไม่ถูกต้อง กรุณาใช้ไฟล์ที่มีโครงสร้างคอลัมน์เหมือนชุดข้อมูลที่ใช้เทรน")
        else:
            st.success(f"✅ อัปโหลดไฟล์สำเร็จ! พบข้อมูลนักเรียนจำนวน {len(df)} รายการ")
            
            # ปุ่มกดยืนยันให้ AI วิเคราะห์
            if st.button("🚀 เริ่มต้นวิเคราะห์ยกชั้นเรียน", type="primary", use_container_width=True):
                with st.spinner("🧠 AI กำลังประมวลผลข้อมูลนักเรียนทั้งหมด..."):
                    try:
                        # แปลง DataFrame เป็น JSON แบบ List of Dicts
                        # ตัดคอลัมน์ G1, G2, G3 ออกถ้ามีติดมาด้วย เพราะตอนใช้งานจริงเรายังไม่รู้เกรด
                        drop_cols = [c for c in ["G1", "G2", "G3", "risk"] if c in df.columns]
                        df_clean = df.drop(columns=drop_cols)
                        
                        # ถ้าไม่มีคอลัมน์ course ให้เติมค่าลงไป
                        if "course" not in df_clean.columns:
                            df_clean["course"] = "math"
                            
                        json_str = df_clean.to_json(orient="records")
                        payload = json.loads(json_str)
                        
                        # ส่งข้อมูลทั้งก้อนไปที่ Endpoint ใหม่
                        response = requests.post("http://127.0.0.1:8000/predict_batch", json=payload)
                        
                        if response.status_code == 200:
                            batch_results = response.json()["batch_results"]
                            res_df = pd.DataFrame(batch_results)
                            
                            st.divider()
                            st.subheader("🎯 สรุปผลการประเมินความเสี่ยง")
                            
                            # นับจำนวนคนเสี่ยง vs คนปกติ
                            high_risk_count = len(res_df[res_df["prediction_class"] == 1])
                            normal_count = len(res_df[res_df["prediction_class"] == 0])
                            
                            m_col1, m_col2, m_col3 = st.columns(3)
                            m_col1.metric("จำนวนนักเรียนทั้งหมด", f"{len(res_df)} คน")
                            m_col2.metric("🟢 ปกติ (Low Risk)", f"{normal_count} คน")
                            m_col3.metric("🔴 เสี่ยงตก (High Risk)", f"{high_risk_count} คน")
                            
                            # แสดงตารางผลลัพธ์
                            st.markdown("### 📋 ตารางรายชื่อนักเรียนที่ต้องเฝ้าระวัง")
                            # ดึงมาโชว์เฉพาะคนที่เสี่ยง
                            risk_table = res_df[res_df["prediction_class"] == 1].copy()
                            risk_table = risk_table.rename(columns={
                                "student_id": "ลำดับที่ (ID)",
                                "risk_status": "สถานะความเสี่ยง",
                                "failures": "ประวัติสอบตก (วิชา)",
                                "absences": "ขาดเรียน (วัน)"
                            })
                            
                            if len(risk_table) > 0:
                                st.dataframe(risk_table.drop(columns=["prediction_class"]), use_container_width=True)
                            else:
                                st.success("🎉 ยินดีด้วย! ไม่พบนักเรียนที่มีความเสี่ยงสูงในข้อมูลชุดนี้")
                                
                            # ปุ่มให้ดาวน์โหลดผลลัพธ์ไปเปิดใน Excel ได้
                            csv_result = res_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 ดาวน์โหลดผลการวิเคราะห์ทั้งหมด (CSV)",
                                data=csv_result,
                                file_name='risk_prediction_results.csv',
                                mime='text/csv',
                            )
                            
                        else:
                            st.error(f"❌ เกิดข้อผิดพลาดจาก API: {response.text}")
                    except Exception as e:
                        st.error(f"🔌 เกิดข้อผิดพลาดในการเชื่อมต่อ: {e}")