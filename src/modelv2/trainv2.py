import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
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

# 4. แบ่งข้อมูล Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. กำหนดค่าสำหรับ Save Model ลงเครื่อง
MODEL_DIR = "StudentCare-AI\src\modelv2\Weight_feature_por"
os.makedirs(MODEL_DIR, exist_ok=True)

# 6. ตั้งค่า MLflow และรันการเทรน
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Weight_feature_por")

with mlflow.start_run():
    # เทรนโมเดล
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # คำนวณ Metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # บันทึก Metrics ลง MLflow
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("RMSE", rmse)
    
    # บันทึก Weights (Feature Importance)
    for name, imp in zip(features, model.feature_importances_):
        mlflow.log_metric(f"weight_{name}", imp)
        
    # บันทึกโมเดลลง MLflow
    mlflow.sklearn.log_model(model, "student_performance_model")
    
    # Save โมเดลลง Folder ในเครื่อง
    joblib.dump(model, os.path.join(MODEL_DIR, "student_model_v2.pkl"))
    
    print("--- Training and Logging Complete ---")
    print(f"R2: {r2:.4f}, RMSE: {rmse:.4f}")
    print(f"Model saved to {MODEL_DIR}/student_model_v2.pkl")