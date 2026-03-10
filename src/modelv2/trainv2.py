import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# 1. โหลดข้อมูล
df = pd.read_csv(r"C:\Ject_MLOps\StudentCare-AI\data\processed\process_for_modelv2\porv2.csv")

# 2. เลือก Features (รวมทั้งวิชาการและสภาพแวดล้อม)
features = [
    'G1', 'G2', 'studytime', 'failures', 'absences',
    'Medu', 'Fedu', 'famrel', 'freetime', 'goout', 
    'nursery', 'internet', 'romantic', 'Dalc', 'Walc'
]
X = df[features].copy()
y = df['G3']

# 3. จัดการ Encoding ข้อมูล Text ให้เป็นตัวเลข (เพื่อให้โมเดลคำนวณได้)
le = LabelEncoder()
cat_cols = ['nursery', 'internet', 'romantic']
for col in cat_cols:
    X[col] = le.fit_transform(X[col])

# 4. แบ่งข้อมูลเป็น 3 ส่วน (Train 70%, Val 15%, Test 15%)
# ขั้นแรกแบ่ง Train และ "Temp" (ที่เหลือ)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# ขั้นสองแบ่ง Temp เป็น Val และ Test อย่างละครึ่ง
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 5. กำหนดค่าสำหรับ Save Model ลงเครื่อง
MODEL_DIR = "StudentCare-AI\src\modelv2\Weight_feature_por_2_1"
os.makedirs(MODEL_DIR, exist_ok=True)

# 6. ตั้งค่า MLflow และรันการเทรน
mlflow.set_tracking_uri("http://localhost:5005")
mlflow.set_experiment("Weight_feature_por_v2_1")

with mlflow.start_run():
    # เทรนโมเดล
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # --- วัดผลบน Validation Set (ใช้ปรับจูน) ---
    y_val_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, y_val_pred)
    mlflow.log_metric("Val_R2", val_r2)
    
    # --- วัดผลบน Test Set (ประสิทธิภาพจริง) ---
    y_test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    mlflow.log_metrics({"Test_MAE": mae, "Test_R2": r2, "Test_RMSE": rmse})
    
    # บันทึก Feature Importance
    for name, imp in zip(features, model.feature_importances_):
        mlflow.log_metric(f"weight_{name}", imp)

    # 7. เช็คประสิทธิภาพ: ลองสุ่มผล Predict เทียบกับค่าจริง 5 แถว
    print("\n--- Model Prediction Check (Test Set) ---")
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred.round(2)})
    print(results_df.head(10))

    # บันทึกโมเดล
    mlflow.sklearn.log_model(model, "student_performance_model")
    joblib.dump(model, os.path.join(MODEL_DIR, "student_model_v2_1.pkl"), protocol=4)
    
    print(f"\nTraining Complete!")
    print(f"Validation R2: {val_r2:.4f}")
    print(f"Test R2: {r2:.4f}")