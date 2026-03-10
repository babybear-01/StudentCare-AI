from google import genai
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import numpy as np
from pathlib import Path
from io import BytesIO

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
# 1.1) SESSION STATE
# ==========================================
if "student_data" not in st.session_state:
    st.session_state.student_data = None

if "result" not in st.session_state:
    st.session_state.result = None

if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = None


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

    .factor-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px 18px;
        margin-bottom: 16px;
    }

    .factor-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 4px;
    }

    .factor-divider {
        color: #94a3b8;
        font-family: monospace;
        margin-bottom: 14px;
        letter-spacing: 0.5px;
    }

    .factor-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        font-size: 1rem;
        margin-bottom: 10px;
        color: #e2e8f0;
        line-height: 1.5;
    }

    .factor-icon {
        font-size: 1.05rem;
        line-height: 1.3;
    }

    .factor-empty {
        color: #bbf7d0;
        font-size: 0.98rem;
        line-height: 1.6;
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
MATH_STEP2_MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "modelv2"
    / "StudentCare-AI"
    / "src"
    / "models"
    / "math_step2_risk_model_v3.pkl"
)

POR_STEP2_MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "modelv2"
    / "StudentCare-AI"
    / "src"
    / "models"
    / "por_step2_risk_model_v3.pkl"
)

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
# 6) STEP 1 DEFAULTS
# ==========================================
STEP1_DEFAULTS = {
    "studytime": 2,
    "failures": 0,
    "absences": 0,
    "Medu": 2,
    "Fedu": 2,
    "famrel": 3,
    "freetime": 3,
    "goout": 3,
    "nursery": 1,
    "internet": 1,
    "romantic": 0,
    "Dalc": 1,
    "Walc": 1,
}

# ==========================================
# 7) LOAD MODELS
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
# 7B) LOAD GEMINI
# ==========================================
@st.cache_resource
def load_gemini_client():

    api_key = st.secrets.get("GEMINI_API_KEY", None)

    if not api_key:
        return None

    client = genai.Client(api_key=api_key)

    return client

# ==========================================
# 7C) CACHE CSV READER (STABLE)
# ==========================================
@st.cache_data(show_spinner=False)
def read_uploaded_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    """
    อ่าน CSV จาก bytes เพื่อให้ cache เสถียรกว่าใช้ uploaded_file object ตรง ๆ
    """
    df = pd.read_csv(BytesIO(file_bytes))
    return df


# ==========================================
# 8) BUSINESS LOGIC
# ==========================================
def normalize_optional_value(value):
    if value == "-" or value == "":
        return np.nan
    return value


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
    processed = {}

    for key, value in student_data.items():
        processed[key] = normalize_optional_value(value)

    binary_map = {
        "ใช่": 1,
        "ไม่ใช่": 0,
        "มี": 1,
        "ไม่มี": 0,
        1: 1,
        0: 0,
    }

    for col in ["nursery", "internet", "romantic"]:
        if col in processed and not pd.isna(processed[col]):
            processed[col] = binary_map.get(processed[col], processed[col])

    return processed

def coerce_numeric_or_nan(value):
    value = normalize_optional_value(value)
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def prepare_step1_input(student_data: dict) -> pd.DataFrame:
    step1_input = preprocess_step1_input(student_data)
    row = {}

    for col in STEP1_FEATURES:
        val = step1_input.get(col, np.nan)

        if pd.isna(val):
            val = STEP1_DEFAULTS.get(col, 0)

        row[col] = val

    step1_df = pd.DataFrame([row], columns=STEP1_FEATURES)

    for col in STEP1_FEATURES:
        step1_df[col] = pd.to_numeric(step1_df[col], errors="coerce")

    for col in STEP1_FEATURES:
        if step1_df[col].isna().any():
            step1_df[col] = step1_df[col].fillna(STEP1_DEFAULTS.get(col, 0))

    return step1_df

def prepare_step2_input(student_data: dict) -> pd.DataFrame:
    row = []

    for col in STEP2_FEATURES:
        val = student_data.get(col, np.nan)
        val = normalize_optional_value(val)
        row.append(val)

    return pd.DataFrame([row], columns=STEP2_FEATURES)


def get_top_risk_factors(student_data: dict):
    factors = []

    g1 = student_data.get("G1", np.nan)
    g2 = student_data.get("G2", np.nan)
    failures = normalize_optional_value(student_data.get("failures", np.nan))
    absences = normalize_optional_value(student_data.get("absences", np.nan))
    studytime = normalize_optional_value(student_data.get("studytime", np.nan))
    goout = normalize_optional_value(student_data.get("goout", np.nan))
    medu = normalize_optional_value(student_data.get("Medu", np.nan))
    walc = normalize_optional_value(student_data.get("Walc", np.nan))

    if not pd.isna(g2):
        if g2 < 10:
            factors.append(("🔴", "G2 ต่ำ", 4))
        elif g2 < 12:
            factors.append(("🟠", "G2 ค่อนข้างต่ำ", 3))

    if not pd.isna(absences):
        if absences >= 10:
            factors.append(("🟠", "ขาดเรียนสูง", 3))
        elif absences >= 5:
            factors.append(("🟡", "ขาดเรียนระดับปานกลาง", 2))

    if not pd.isna(studytime):
        if studytime == 1:
            factors.append(("🟡", "เวลาเรียนต่ำ", 2))
        elif studytime == 2:
            factors.append(("🟡", "เวลาเรียนปานกลาง", 1))

    if not pd.isna(failures):
        if failures >= 2:
            factors.append(("🔴", "มีประวัติสอบตกหลายครั้ง", 4))
        elif failures == 1:
            factors.append(("🟠", "มีประวัติสอบตก", 3))

    if not pd.isna(g1):
        if g1 < 10:
            factors.append(("🟠", "G1 ต่ำ", 3))
        elif g1 < 12:
            factors.append(("🟡", "G1 ค่อนข้างต่ำ", 2))

    if not pd.isna(goout):
        if goout >= 4:
            factors.append(("🟡", "ออกไปเที่ยวบ่อย", 1))

    if not pd.isna(medu):
        if medu == 0:
            factors.append(("🟡", "การศึกษาของแม่ต่ำมาก", 1))
        elif medu == 1:
            factors.append(("🟡", "การศึกษาของแม่ค่อนข้างต่ำ", 1))

    if not pd.isna(walc):
        if walc >= 4:
            factors.append(("🟡", "การดื่มวันหยุดค่อนข้างสูง", 1))

    factors = sorted(factors, key=lambda x: x[2], reverse=True)
    return factors[:3]


def render_top_risk_factors_ai_style(student_data: dict):
    st.markdown("### 🔎 Top Risk Factors")

    top_factors = get_top_risk_factors(student_data)

    if len(top_factors) > 0:
        factor_lines = ""
        for icon, label, _ in top_factors:
            factor_lines += f"""
<div class="factor-item">
    <span class="factor-icon">{icon}</span>
    <span>{label}</span>
</div>
"""

        card_html = f"""
<div class="factor-card">
    <div class="factor-title">Top Risk Factors</div>
    <div class="factor-divider">━━━━━━━━━━━━━━━━━━</div>
    {factor_lines}
</div>
"""
        st.markdown(card_html, unsafe_allow_html=True)

    else:
        st.markdown(
            """
<div class="factor-card">
    <div class="factor-title">Top Risk Factors</div>
    <div class="factor-divider">━━━━━━━━━━━━━━━━━━</div>
    <div class="factor-empty">🟢 ไม่พบปัจจัยเสี่ยงเด่นชัดจากข้อมูลที่กรอก</div>
</div>
""",
            unsafe_allow_html=True
        )


def predict_single(student_data: dict, subject: str):
    # ---------- STEP 1 ----------
    step1_model = load_step1_model(subject)
    step1_df = prepare_step1_input(student_data)

    predicted_g3 = float(step1_model.predict(step1_df)[0])
    academic_status = classify_g3(predicted_g3)
    risk_zone = create_risk_zone(predicted_g3)

    # ---------- STEP 2B ----------
    step2_model = load_step2_model(subject)
    step2_df = prepare_step2_input(student_data)

    raw_pred = step2_model.predict(step2_df)[0]

    if isinstance(raw_pred, str):
        risk_label = raw_pred
    else:
        try:
            risk_label = CLASS_MAP.get(int(raw_pred), str(raw_pred))
        except Exception:
            risk_label = str(raw_pred)

    risk_probabilities = None
    if hasattr(step2_model, "predict_proba"):
        probs = step2_model.predict_proba(step2_df)[0]

        try:
            if hasattr(step2_model, "named_steps") and "classifier" in step2_model.named_steps:
                raw_classes = step2_model.named_steps["classifier"].classes_
            else:
                raw_classes = step2_model.classes_

            risk_probabilities = {}
            for cls, prob in zip(raw_classes, probs):
                if isinstance(cls, str):
                    risk_probabilities[cls] = float(prob)
                else:
                    risk_probabilities[CLASS_MAP.get(int(cls), str(cls))] = float(prob)
        except Exception:
            risk_probabilities = None

    return {
        "subject": subject,
        "predicted_g3": predicted_g3,
        "academic_status": academic_status,
        "risk_zone": risk_zone,
        "risk_label": risk_label,
        "risk_probabilities": risk_probabilities,
    }


def predict_batch(df: pd.DataFrame, default_subject: str = "math"):
    df = df.copy()
    results = []

    for _, row in df.iterrows():
        subject = row.get("course", default_subject)
        if subject not in ["math", "por"]:
            subject = default_subject

        student_data = {}
        for col in STEP1_FEATURES:
            value = row[col] if col in row else np.nan
            if pd.isna(value):
                value = np.nan
            student_data[col] = value

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


def generate_ai_summary(student_data: dict, result: dict):

    client = load_gemini_client()

    if client is None:
        return "ไม่พบ Gemini API Key"

     # ✅ สร้าง factor_text ก่อนใช้
    top_factors = get_top_risk_factors(student_data)
    factor_text = ", ".join([label for _, label, _ in top_factors]) if top_factors else "ไม่พบปัจจัยเสี่ยงเด่นชัด"

    prompt = f"""
วิเคราะห์ความเสี่ยงทางการเรียนของนักเรียนเพื่อช่วยให้อาจารย์ติดตามผลการเรียน

ห้ามแต่งข้อมูลเพิ่ม ให้ใช้เฉพาะข้อมูลที่ให้

Student Data
G1: {student_data.get("G1")}
G2: {student_data.get("G2")}
Failures: {student_data.get("failures")}
Absences: {student_data.get("absences")}
StudyTime: {student_data.get("studytime")}
GoOut: {student_data.get("goout")}

Model Result
Predicted G3: {result.get("predicted_g3")}
Risk Level: {result.get("risk_label")}

Top Factors
{factor_text}

ตอบเป็นภาษาไทย กระชับ ไม่เกิน 6 บรรทัด

รูปแบบคำตอบ:

📊 ภาพรวมผลการเรียน:
<คำอธิบาย>

⚠️ ปัจจัยเสี่ยงสำคัญ:
<คำอธิบาย>

👩‍🏫 ข้อเสนอแนะสำหรับอาจารย์:
<คำอธิบาย>
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"Gemini error: {e}"
    



# ==========================================
# 9) SIDEBAR
# ==========================================
with st.sidebar:
    st.title("🎓 StudentCare-AI")
    st.markdown("ระบบทำนายคะแนนสอบปลายภาค(G3)และจัดระดับความเสี่ยงของนักเรียนแบบรายบุคคลและแบบยกชั้นเรียน")
    st.divider()

    app_mode = st.radio(
        "เลือกโหมดการทำงาน",
        ["👤 วิเคราะห์รายบุคคล", "📁 ประเมินยกชั้นเรียน"]
    )

    st.divider()
    st.caption("KMITL Student Project")

# ==========================================
# 10) INDIVIDUAL MODE
# ==========================================
if app_mode == "👤 วิเคราะห์รายบุคคล":
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">🎓 Student Risk Analyzer</div>
            <div class="hero-subtitle">
                กรอกข้อมูลนักเรียนเพื่อทำนายคะแนนสอบปลายภาค(G3)และประเมินระดับความเสี่ยงแบบรายบุคคล
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.error(
        """
⚠️ **คำแนะนำในการกรอกข้อมูล**

หากเลือกค่า **"-" (ไม่ทราบข้อมูล)** อาจทำให้ **ผลการทำนายมีความแม่นยำน้อยลง**
เพื่อให้ AI วิเคราะห์ได้ดีที่สุด ควรกรอกข้อมูลให้ครบมากที่สุด
        """
    )

    with st.form("individual_form"):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📌 ข้อมูลที่จำเป็น</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        subject = c1.selectbox("เลือกรายวิชา", ["math", "por"])
        G1 = c2.number_input("คะแนนสอบช่วงที่ 1 (0-20)", min_value=0.0, max_value=20.0, value=10.0)
        G2 = c3.number_input("คะแนนสอบช่วงที่ 2 (0-20)", min_value=0.0, max_value=20.0, value=10.0)
        absences = c4.number_input("จำนวนครั้งที่ขาดเรียน", min_value=0, max_value=93, value=0)

        failures = st.selectbox("จำนวนครั้งที่สอบตกในอดีต", [0, 1, 2, 3, 4], index=0)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔍 ข้อมูลเพิ่มเติม (เลือก "-" ได้ถ้าไม่ทราบ)</div>', unsafe_allow_html=True)

        c5, c6, c7 = st.columns(3)

        study_options = {
            "-": "-",
            1: "น้อยกว่า 2 ชั่วโมง/สัปดาห์",
            2: "2-5 ชั่วโมง/สัปดาห์",
            3: "5-10 ชั่วโมง/สัปดาห์",
            4: "มากกว่า 10 ชั่วโมง/สัปดาห์",
        }
        studytime_txt = c5.selectbox("เวลาที่ใช้ในการเรียนต่อสัปดาห์", list(study_options.values()), index=0)
        studytime = [k for k, v in study_options.items() if v == studytime_txt][0]

        edu_options = {
            "-": "-",
            0: "ไม่มีการศึกษา",
            1: "ประถมศึกษา",
            2: "มัธยมต้น",
            3: "มัธยมปลาย",
            4: "อุดมศึกษา / มหาวิทยาลัย",
        }
        Medu_txt = c6.selectbox("ระดับการศึกษาของแม่", list(edu_options.values()), index=0)
        Medu = [k for k, v in edu_options.items() if v == Medu_txt][0]

        Fedu_txt = c7.selectbox("ระดับการศึกษาของพ่อ", list(edu_options.values()), index=0)
        Fedu = [k for k, v in edu_options.items() if v == Fedu_txt][0]

        c8, c9, c10 = st.columns(3)

        famrel_options = {
            "-": "-",
            1: "แย่มาก",
            2: "ค่อนข้างแย่",
            3: "ปานกลาง",
            4: "ค่อนข้างดี",
            5: "ดีมาก",
        }
        famrel_txt = c8.selectbox("ความสัมพันธ์ในครอบครัว", list(famrel_options.values()), index=0)
        famrel = [k for k, v in famrel_options.items() if v == famrel_txt][0]

        level_options = {
            "-": "-",
            1: "ต่ำมาก",
            2: "ต่ำ",
            3: "ปานกลาง",
            4: "สูง",
            5: "มากที่สุด",
        }
        freetime_txt = c9.selectbox("เวลาว่างหลังเลิกเรียน", list(level_options.values()), index=0)
        freetime = [k for k, v in level_options.items() if v == freetime_txt][0]

        goout_txt = c10.selectbox("ความถี่ในการไปเที่ยวกับเพื่อน", list(level_options.values()), index=0)
        goout = [k for k, v in level_options.items() if v == goout_txt][0]

        c11, c12, c13 = st.columns(3)

        Dalc_txt = c11.selectbox("การดื่มแอลกอฮอล์ในวันธรรมดา", list(level_options.values()), index=0)
        Dalc = [k for k, v in level_options.items() if v == Dalc_txt][0]

        Walc_txt = c12.selectbox("การดื่มแอลกอฮอล์ในวันหยุด", list(level_options.values()), index=0)
        Walc = [k for k, v in level_options.items() if v == Walc_txt][0]

        nursery_txt = c13.selectbox("เคยเรียนอนุบาลหรือไม่", ["-", "ใช่", "ไม่ใช่"], index=0)

        c14, c15 = st.columns(2)
        internet_txt = c14.selectbox("มี Internet หรือไม่", ["-", "มี", "ไม่มี"], index=0)
        romantic_txt = c15.selectbox("มีแฟนหรือไม่", ["-", "มี", "ไม่มี"], index=0)

        st.markdown("</div>", unsafe_allow_html=True)

        submit = st.form_submit_button("🔍 วิเคราะห์ข้อมูลนักเรียน", use_container_width=True)

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
        st.session_state.student_data = student_data
        st.session_state.result = result
        st.session_state.ai_summary = None
    if st.session_state.result is not None:
        student_data = st.session_state.student_data
        result = st.session_state.result

        st.divider()
        render_top_risk_factors_ai_style(student_data)

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
            st.plotly_chart(fig, use_container_width=True)

            cc1, cc2 = st.columns(2)
            cc1.metric("Subject", result["subject"])
            cc2.metric("Predicted G3", f"{result['predicted_g3']:.2f}")

            if result["academic_status"] == "Normal":
                st.success(f"Academic Status: {result['academic_status']}")
            elif result["academic_status"] == "Watchlist":
                st.warning(f"Academic Status: {result['academic_status']}")
            else:
                st.error(f"Academic Status: {result['academic_status']}")

            st.info(f"Risk Zone: {result['risk_zone']}")
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
                st.bar_chart(prob_df.set_index("Risk"), use_container_width=True)

                st.write(f"- High Risk: {result['risk_probabilities'].get('High Risk', 0.0):.4f}")
                st.write(f"- Medium Risk: {result['risk_probabilities'].get('Medium Risk', 0.0):.4f}")
                st.write(f"- Low Risk: {result['risk_probabilities'].get('Low Risk', 0.0):.4f}")

            st.markdown("</div>", unsafe_allow_html=True)
            if st.button("✨ ขอสรุปผลและคำแนะนำเพิ่มเติม", use_container_width=True):
                with st.spinner("กำลังให้ AI วิเคราะห์..."):
                    st.session_state.ai_summary = generate_ai_summary(student_data, result)

            if st.session_state.ai_summary:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.subheader("สรุปผลและคำแนะจาก AI")
                formatted_summary = st.session_state.ai_summary.replace("\n", "<br>")
                st.markdown(formatted_summary, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 11) BATCH MODE
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

        @st.cache_data
        def read_uploaded_csv(uploaded_file):
            df = pd.read_csv(uploaded_file)
            return df
        st.write("ตัวอย่างข้อมูล")
        st.dataframe(df.head(), use_container_width=True)

        required_cols = set(STEP1_FEATURES)
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            st.error(f"ไฟล์ขาดคอลัมน์: {missing}")
        else:
            if st.button("🚀 เริ่มวิเคราะห์", use_container_width=True):
                result_df = predict_batch(df, default_subject=default_subject)

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
                st.bar_chart(summary_df.set_index("Risk_Label"), use_container_width=True)

                st.subheader("ผลลัพธ์รายคน")
                st.dataframe(result_df, use_container_width=True)

                csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 ดาวน์โหลดผลลัพธ์ CSV",
                    data=csv_data,
                    file_name="studentcare_ai_results.csv",
                    mime="text/csv"
                )