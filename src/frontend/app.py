import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from pathlib import Path

# ==========================================
# 1) PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="StudentCare-AI | Student Risk Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2) CUSTOM CSS
# ==========================================
st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1.0rem;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.0rem;
        padding-bottom: 2rem;
    }

    .hero-box {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #cbd5e1;
        margin-bottom: 0;
    }

    .section-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.65rem 1rem;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.8rem;
    }

    .result-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }

    .risk-chip {
        display: inline-block;
        padding: 0.38rem 0.85rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 0.2rem;
        margin-bottom: 0.6rem;
    }

    .risk-high {
        background: rgba(239,68,68,0.18);
        color: #fecaca;
        border: 1px solid rgba(239,68,68,0.35);
    }

    .risk-medium {
        background: rgba(245,158,11,0.18);
        color: #fde68a;
        border: 1px solid rgba(245,158,11,0.35);
    }

    .risk-low {
        background: rgba(34,197,94,0.18);
        color: #bbf7d0;
        border: 1px solid rgba(34,197,94,0.35);
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: -0.2rem;
    }

    div[data-testid="stForm"] {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1rem 1rem 0.3rem 1rem;
        background: rgba(255,255,255,0.02);
    }

    button[kind="primary"] {
        border-radius: 12px !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 3) PATHS
# ==========================================
CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = None
for parent in CURRENT_FILE.parents:
    if (parent / "data").exists() and (parent / "src").exists():
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError("Could not find project root directory")

# Step 1 models (predict G3)
MATH_STEP1_MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "modelv2"
    / "StudentCare-AI"
    / "src"
    / "modelv2"
    / "Weight_feature_mat_2"
    / "student_model_v2.pkl"
)

POR_STEP1_MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "modelv2"
    / "StudentCare-AI"
    / "src"
    / "modelv2"
    / "Weight_feature_por_2"
    / "student_model_v2.pkl"
)

# Step 2B models (predict risk label)
MATH_STEP2_MODEL_PATH = PROJECT_ROOT / "src" / "models" / "math_step2_risk_model.pkl"
POR_STEP2_MODEL_PATH = PROJECT_ROOT / "src" / "models" / "por_step2_risk_model.pkl"

# ==========================================
# 4) FEATURES
# ==========================================
STEP1_FEATURES = [
    "G1",
    "G2",
    "studytime",
    "failures",
    "absences",
    "Medu",
    "Fedu",
    "famrel",
    "freetime",
    "goout",
    "nursery",
    "internet",
    "romantic",
    "Dalc",
    "Walc",
]

STEP2_FEATURES = [
    "G1",
    "G2",
    "failures",
    "absences",
    "studytime",
    "goout",
    "Medu",
    "Walc",
]

BATCH_REQUIRED_COLUMNS = STEP1_FEATURES + ["course"]

# ==========================================
# 5) CLASS MAP
# ==========================================
CLASS_MAP = {
    0: "High Risk",
    1: "Low Risk",
    2: "Medium Risk",
}

# ==========================================
# 6) LOAD MODELS
# ==========================================
@st.cache_resource
def load_step1_model(subject: str):
    if subject == "math":
        if not MATH_STEP1_MODEL_PATH.exists():
            raise FileNotFoundError(f"Math Step 1 model not found: {MATH_STEP1_MODEL_PATH}")
        return joblib.load(MATH_STEP1_MODEL_PATH)

    if subject == "por":
        if not POR_STEP1_MODEL_PATH.exists():
            raise FileNotFoundError(f"Portuguese Step 1 model not found: {POR_STEP1_MODEL_PATH}")
        return joblib.load(POR_STEP1_MODEL_PATH)

    raise ValueError("subject must be 'math' or 'por'")


@st.cache_resource
def load_step2_model(subject: str):
    if subject == "math":
        if not MATH_STEP2_MODEL_PATH.exists():
            raise FileNotFoundError(f"Math Step 2 model not found: {MATH_STEP2_MODEL_PATH}")
        return joblib.load(MATH_STEP2_MODEL_PATH)

    if subject == "por":
        if not POR_STEP2_MODEL_PATH.exists():
            raise FileNotFoundError(f"Portuguese Step 2 model not found: {POR_STEP2_MODEL_PATH}")
        return joblib.load(POR_STEP2_MODEL_PATH)

    raise ValueError("subject must be 'math' or 'por'")


# ==========================================
# 7) BUSINESS LOGIC
# ==========================================
def classify_g3(predicted_g3: float) -> str:
    if predicted_g3 < 10:
        return "Academic Risk"
    elif predicted_g3 < 12:
        return "Watchlist"
    return "Normal"


def create_risk_zone(predicted_g3: float) -> str:
    if predicted_g3 < 10:
        return "High Risk Zone"
    elif predicted_g3 < 12:
        return "Medium Risk Zone"
    return "Low Risk Zone"


def preprocess_step1_input(student_data: dict) -> dict:
    processed = student_data.copy()
    binary_map = {"yes": 1, "no": 0, 1: 1, 0: 0}

    for col in ["nursery", "internet", "romantic"]:
        if col in processed:
            processed[col] = binary_map.get(processed[col], processed[col])

    return processed


def get_risk_badge(label: str) -> str:
    if label == "High Risk":
        return "🔴 High Risk"
    if label == "Medium Risk":
        return "🟠 Medium Risk"
    if label == "Low Risk":
        return "🟢 Low Risk"
    return str(label)


def predict_single(student_data: dict, subject: str):
    # ---------- STEP 1 ----------
    step1_model = load_step1_model(subject)
    step1_input = preprocess_step1_input(student_data)

    step1_df = pd.DataFrame(
        [[step1_input[col] for col in STEP1_FEATURES]],
        columns=STEP1_FEATURES
    )

    predicted_g3 = float(step1_model.predict(step1_df)[0])
    academic_status = classify_g3(predicted_g3)
    risk_zone = create_risk_zone(predicted_g3)

    # ---------- STEP 2B ----------
    step2_model = load_step2_model(subject)

    step2_df = pd.DataFrame(
        [[student_data[col] for col in STEP2_FEATURES]],
        columns=STEP2_FEATURES
    )

    raw_label = int(step2_model.predict(step2_df)[0])
    risk_label = CLASS_MAP.get(raw_label, "Unknown")

    risk_probabilities = None
    if hasattr(step2_model, "predict_proba"):
        probs = step2_model.predict_proba(step2_df)[0]
        classifier = step2_model.named_steps["classifier"]
        raw_classes = classifier.classes_

        risk_probabilities = {
            CLASS_MAP.get(int(cls), str(cls)): float(prob)
            for cls, prob in zip(raw_classes, probs)
        }

    return {
        "subject": subject,
        "predicted_g3": predicted_g3,
        "academic_status": academic_status,
        "risk_zone": risk_zone,
        "risk_label": risk_label,
        "risk_probabilities": risk_probabilities,
    }


def predict_batch(df: pd.DataFrame):
    results = []

    for _, row in df.iterrows():
        subject = row.get("course", "math")
        if subject not in ["math", "por"]:
            subject = "math"

        student_data = {col: row[col] for col in STEP1_FEATURES}
        result = predict_single(student_data, subject)

        prob_high = None
        prob_medium = None
        prob_low = None

        if result["risk_probabilities"] is not None:
            prob_high = round(result["risk_probabilities"].get("High Risk", 0.0), 4)
            prob_medium = round(result["risk_probabilities"].get("Medium Risk", 0.0), 4)
            prob_low = round(result["risk_probabilities"].get("Low Risk", 0.0), 4)

        results.append({
            "Predicted_G3": round(result["predicted_g3"], 2),
            "Academic_Status": result["academic_status"],
            "Risk_Zone": result["risk_zone"],
            "Risk_Label": result["risk_label"],
            "Prob_High_Risk": prob_high,
            "Prob_Medium_Risk": prob_medium,
            "Prob_Low_Risk": prob_low,
        })

    result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]
    return result_df


# ==========================================
# 8) SIDEBAR
# ==========================================
with st.sidebar:
    st.title("🎓 StudentCare-AI")
    st.markdown("ระบบทำนาย G3 และจัดระดับความเสี่ยงของนักเรียน")
    st.divider()

    app_mode = st.radio(
        "เลือกโหมดการทำงาน",
        ["👤 วิเคราะห์รายบุคคล", "📁 ประเมินยกชั้นเรียน"]
    )

    st.divider()
    st.caption("KMITL Student Project")


# ==========================================
# 9) INDIVIDUAL MODE
# ==========================================
if app_mode == "👤 วิเคราะห์รายบุคคล":
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">🎓 Student Risk Analyzer</div>
            <div class="hero-subtitle">
                กรอกข้อมูลนักเรียนเพื่อทำนายคะแนน G3 และประเมินระดับความเสี่ยงแบบรายบุคคล
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("individual_form"):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📘 ข้อมูลวิชาและผลการเรียน</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        subject = c1.selectbox("เลือกรายวิชา", ["math", "por"])
        G1 = c2.number_input("G1", min_value=0.0, max_value=20.0, value=10.0)
        G2 = c3.number_input("G2", min_value=0.0, max_value=20.0, value=10.0)

        c4, c5, c6 = st.columns(3)
        studytime = c4.selectbox("studytime (1-4)", [1, 2, 3, 4], index=1)
        failures = c5.number_input("failures", min_value=0, max_value=4, value=0)
        absences = c6.number_input("absences", min_value=0, max_value=93, value=0)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👨‍👩‍👧 ปัจจัยครอบครัวและพฤติกรรม</div>', unsafe_allow_html=True)

        c7, c8, c9 = st.columns(3)
        Medu = c7.selectbox("Medu (0-4)", [0, 1, 2, 3, 4], index=2)
        Fedu = c8.selectbox("Fedu (0-4)", [0, 1, 2, 3, 4], index=2)
        famrel = c9.selectbox("famrel (1-5)", [1, 2, 3, 4, 5], index=3)

        c10, c11, c12 = st.columns(3)
        freetime = c10.selectbox("freetime (1-5)", [1, 2, 3, 4, 5], index=2)
        goout = c11.selectbox("goout (1-5)", [1, 2, 3, 4, 5], index=1)
        Dalc = c12.selectbox("Dalc (1-5)", [1, 2, 3, 4, 5], index=0)

        c13, c14, c15 = st.columns(3)
        Walc = c13.selectbox("Walc (1-5)", [1, 2, 3, 4, 5], index=0)
        nursery_txt = c14.selectbox("nursery", ["yes", "no"], index=0)
        internet_txt = c15.selectbox("internet", ["yes", "no"], index=0)

        romantic_txt = st.selectbox("romantic", ["yes", "no"], index=1)
        st.markdown("</div>", unsafe_allow_html=True)

        submit = st.form_submit_button("🔍 วิเคราะห์ข้อมูลนักเรียน", width="stretch")

    if submit:
        student_data = {
            "G1": G1,
            "G2": G2,
            "studytime": studytime,
            "failures": failures,
            "absences": absences,
            "Medu": Medu,
            "Fedu": Fedu,
            "famrel": famrel,
            "freetime": freetime,
            "goout": goout,
            "nursery": nursery_txt,
            "internet": internet_txt,
            "romantic": romantic_txt,
            "Dalc": Dalc,
            "Walc": Walc,
        }

        result = predict_single(student_data, subject)

        st.divider()
        left, right = st.columns([1.05, 1.1], gap="large")

        with left:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("Predicted G3")

            gauge_value = max(0, min(20, result["predicted_g3"]))
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_value,
                number={"suffix": " / 20"},
                gauge={
                    "axis": {"range": [0, 20]},
                    "bar": {"color": "#16a34a"},
                    "steps": [
                        {"range": [0, 10], "color": "#fecaca"},
                        {"range": [10, 12], "color": "#fef3c7"},
                        {"range": [12, 20], "color": "#d9f99d"},
                    ],
                },
                title={"text": "Predicted G3"}
            ))
            st.plotly_chart(fig, width="stretch")

            c1, c2 = st.columns(2)
            c1.metric("Subject", result["subject"])
            c2.metric("Predicted G3", f"{result['predicted_g3']:.2f}")

            if result["academic_status"] == "Normal":
                st.success(f"Academic Status: {result['academic_status']}")
            elif result["academic_status"] == "Watchlist":
                st.warning(f"Academic Status: {result['academic_status']}")
            else:
                st.error(f"Academic Status: {result['academic_status']}")

            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("ผลวิเคราะห์")

            st.write("**Student Risk Level:**")

            if result["risk_label"] == "High Risk":
                st.markdown('<span class="risk-chip risk-high">🔴 High Risk</span>', unsafe_allow_html=True)
                st.error("นักเรียนอยู่ในกลุ่มความเสี่ยงสูง ควรติดตามอย่างใกล้ชิด")
            elif result["risk_label"] == "Medium Risk":
                st.markdown('<span class="risk-chip risk-medium">🟠 Medium Risk</span>', unsafe_allow_html=True)
                st.warning("นักเรียนอยู่ในกลุ่มเฝ้าระวัง ควรมีการติดตามต่อเนื่อง")
            else:
                st.markdown('<span class="risk-chip risk-low">🟢 Low Risk</span>', unsafe_allow_html=True)
                st.success("นักเรียนอยู่ในกลุ่มความเสี่ยงต่ำ")

            if result["risk_probabilities"] is not None:
                st.markdown("### Risk Probabilities")
                prob_df = pd.DataFrame({
                    "Risk": ["High Risk", "Medium Risk", "Low Risk"],
                    "Probability": [
                        result["risk_probabilities"].get("High Risk", 0.0),
                        result["risk_probabilities"].get("Medium Risk", 0.0),
                        result["risk_probabilities"].get("Low Risk", 0.0),
                    ]
                })
                st.bar_chart(prob_df.set_index("Risk"), width="stretch")

                st.write(f"- High Risk: {result['risk_probabilities'].get('High Risk', 0.0):.4f}")
                st.write(f"- Medium Risk: {result['risk_probabilities'].get('Medium Risk', 0.0):.4f}")
                st.write(f"- Low Risk: {result['risk_probabilities'].get('Low Risk', 0.0):.4f}")

            st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# 10) BATCH MODE
# ==========================================
elif app_mode == "📁 ประเมินยกชั้นเรียน":
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">📁 Classroom Risk Evaluation</div>
            <div class="hero-subtitle">
                อัปโหลดไฟล์ CSV เพื่อประเมินผลนักเรียนหลายคนพร้อมกัน และดูภาพรวมทั้งชั้นเรียน
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    default_subject = st.selectbox(
        "เลือกรายวิชาสำหรับไฟล์นี้",
        ["math", "por"],
        index=0
    )

    st.markdown("**คอลัมน์ที่จำเป็น**")
    st.code(", ".join(BATCH_REQUIRED_COLUMNS), language="text")

    uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("ตัวอย่างข้อมูล")
        st.dataframe(df.head(), width="stretch")

        required_cols = set(STEP1_FEATURES)
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            st.error(f"ไฟล์ขาดคอลัมน์: {missing}")
        else:
            if "course" not in df.columns:
                df["course"] = default_subject
                st.warning(f"ไม่พบคอลัมน์ course, ระบบตั้งค่าเป็น {default_subject} ให้ทั้งหมด")

            if st.button("🚀 เริ่มวิเคราะห์", width="stretch"):
                result_df = predict_batch(df)

                st.divider()
                st.subheader("สรุปผล")

                low_count = (result_df["Risk_Label"] == "Low Risk").sum()
                medium_count = (result_df["Risk_Label"] == "Medium Risk").sum()
                high_count = (result_df["Risk_Label"] == "High Risk").sum()

                m1, m2, m3 = st.columns(3)
                m1.metric("Low Risk", int(low_count))
                m2.metric("Medium Risk", int(medium_count))
                m3.metric("High Risk", int(high_count))

                st.subheader("Risk Distribution Chart")
                summary_df = pd.DataFrame({
                    "Risk_Label": ["Low Risk", "Medium Risk", "High Risk"],
                    "Count": [int(low_count), int(medium_count), int(high_count)]
                })
                st.bar_chart(summary_df.set_index("Risk_Label"), width="stretch")

                st.subheader("ผลลัพธ์รายคน")
                st.dataframe(result_df, width="stretch")

                csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 ดาวน์โหลดผลลัพธ์ CSV",
                    data=csv_data,
                    file_name="studentcare_ai_results.csv",
                    mime="text/csv"
                )