import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
from typing import List

# ==========================================
# 1. โหลดโมเดลที่ดีที่สุดจาก MLflow อัตโนมัติ
# ==========================================
EXPERIMENT_NAME = "Student_Risk_Prediction"

def load_best_model():
    client = MlflowClient()
    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise ValueError(f"ไม่พบ Experiment ชื่อ {EXPERIMENT_NAME}")
            
        # หา Run ที่ได้คะแนน f1_score สูงสุด
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.f1_score DESC"],
            max_results=1
        )
        
        if not runs:
            raise ValueError("ยังไม่มีการเทรนโมเดลใน MLflow")
            
        best_run_id = runs[0].info.run_id
        print(f"✅ Loaded Best Model automatically from Run ID: {best_run_id}")
        
        # โหลดตัว Pipeline มาใช้งาน
        model_uri = f"runs:/{best_run_id}/random_forest_pipeline"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# โหลดโมเดลเตรียมไว้ตอนเปิดเซิร์ฟเวอร์
model = load_best_model()

# ==========================================
# 2. กำหนดหน้าตาของข้อมูลที่จะรับเข้ามา (Schema)
# ==========================================
class StudentData(BaseModel):
    school: str = "GP"
    sex: str = "F"
    age: int = 16
    address: str = "U"
    famsize: str = "GT3"
    Pstatus: str = "T"
    Medu: int = 4
    Fedu: int = 4
    Mjob: str = "teacher"
    Fjob: str = "health"
    reason: str = "course"
    guardian: str = "mother"
    traveltime: int = 1
    studytime: int = 2
    failures: int = 0
    schoolsup: str = "no"
    famsup: str = "yes"
    paid: str = "no"
    activities: str = "yes"
    nursery: str = "yes"
    higher: str = "yes"
    internet: str = "yes"
    romantic: str = "no"
    famrel: int = 4
    freetime: int = 3
    goout: int = 2
    Dalc: int = 1
    Walc: int = 1
    health: int = 5
    absences: int = 0
    course: str = "math"

# ==========================================
# 3. สร้าง FastAPI App และ Endpoint
# ==========================================
app = FastAPI(
    title="Student Care Risk Prediction API",
    description="API สำหรับประเมินความเสี่ยงเด็กนักเรียนที่มีโอกาสสอบตก",
    version="1.0.0"
)

@app.get("/")
def home():
    return {"message": "Welcome to Student Care API. Model is loaded and ready."}

@app.post("/predict")
def predict_risk(student: StudentData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    try:
        # แปลงข้อมูล JSON จาก User ให้เป็น DataFrame 1 แถว
        input_data = student.model_dump()
        df = pd.DataFrame([input_data])
        
        # ให้โมเดลทำนาย (Pipeline จะจัดการ Preprocess ให้เอง)
        prediction_array = model.predict(df)
        prediction = int(prediction_array[0])
        
        risk_status = "High Risk (เสี่ยงตก)" if prediction == 1 else "Normal (ปกติ)"
        
        return {
            "prediction_class": prediction,
            "risk_status": risk_status,
            "student_profile": {
                "age": student.age, 
                "failures": student.failures, 
                "absences": student.absences
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/predict_batch")
def predict_risk_batch(students: List[StudentData]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    try:
        # แปลงข้อมูล JSON หลายๆ คน ให้เป็น DataFrame
        input_data = [student.model_dump() for student in students]
        df = pd.DataFrame(input_data)
        
        # ทำนายผลทีเดียวทั้งตาราง
        predictions = model.predict(df)
        
        results = []
        for i, pred in enumerate(predictions):
            pred_int = int(pred)
            risk_status = "High Risk" if pred_int == 1 else "Normal"
            
            # 💡 แก้ไขตรงนี้: ใช้ int() ครอบค่าเพื่อป้องกัน Error จาก Numpy
            results.append({
                "student_id": i + 1,
                "prediction_class": pred_int,
                "risk_status": risk_status,
                "failures": int(df.iloc[i]["failures"]), 
                "absences": int(df.iloc[i]["absences"])
            })
            
        return {"batch_results": results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))