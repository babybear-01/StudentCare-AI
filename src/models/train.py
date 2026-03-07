from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split, GridSearchCV

# นำเข้า MLflow สำหรับ MLOps Tracking
import mlflow
import mlflow.sklearn

from src.data.preprocess import load_combined_dataset, make_label, split_xy

# =====================================
# CONFIG
# =====================================
TEST_SIZE = 0.2
RANDOM_STATE = 42
PRED_THRESHOLD = 0.5
EXPERIMENT_NAME = "Student_Risk_Prediction"

@dataclass
class TrainingBundle:
    X_train_raw: pd.DataFrame
    X_val_raw: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray


def prepare_data(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> TrainingBundle:
    """โหลดและแบ่งข้อมูล"""
    df = load_combined_dataset()
    df = make_label(df)

    X, y = split_xy(df)

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return TrainingBundle(
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        y_train=np.asarray(y_train),
        y_val=np.asarray(y_val),
    )


def build_pipeline(X_sample: pd.DataFrame) -> Pipeline:
    """สร้างท่อส่งข้อมูล (Pipeline) รวม Preprocessing และ Model"""
    cat_cols = X_sample.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_sample.columns if c not in cat_cols]

    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()), 
    ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ],
        remainder="drop",
    )

    # สร้างโมเดลเปล่าๆ ไว้ก่อน เดี๋ยว GridSearchCV จะจัดการใส่ Hyperparameters ให้เอง
    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    return full_pipeline


def main():
    print("🚀 TRAINING PIPELINE STARTED (MLOps Version with MLflow)")
    print("=" * 80)
    
    # 1. ตั้งชื่อ Project (Experiment) ใน MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = prepare_data()
    pipeline = build_pipeline(data.X_train_raw)
    
    # 2. เริ่มบันทึกการทดลองลง MLflow
    # เราสามารถรันสคริปต์นี้กี่รอบก็ได้ MLflow จะจำให้ว่ารอบไหนดียังไง
    with mlflow.start_run(run_name="RandomForest_Tuning"):
        
        print("🔍 Starting Hyperparameter Tuning (GridSearchCV)...")
        train_start_time = time.time()
        
        # กำหนดค่าที่ต้องการให้ AI ลองจูนหาค่าที่ดีที่สุด (Hyperparameter Grid)
        param_grid = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [5, 10, 15],
            "classifier__min_samples_split": [2, 5]
        }
        
        # ใช้ GridSearchCV เพื่อหาโมเดลที่ให้ F1-Score สำหรับดักจับคนตก (Class 1) ดีที่สุด
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=3, 
            scoring="f1", 
            n_jobs=-1,
            verbose=1
        )
        
        # เทรนและหาค่าที่ดีที่สุด
        grid_search.fit(data.X_train_raw, data.y_train)
        
        total_training_time = time.time() - train_start_time
        print(f"✅ Tuning completed in {total_training_time:.2f} seconds.")
        
        # ดึงโมเดลตัวที่ชนะมาใช้
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print("🏆 Best Parameters Found:")
        for k, v in best_params.items():
            print(f"  - {k}: {v}")
            
        # 3. ประเมินผลกับข้อมูล Validation
        print("\n📊 Evaluating on Validation Set...")
        val_proba = best_model.predict_proba(data.X_val_raw)[:, 1]
        y_pred = (val_proba >= PRED_THRESHOLD).astype(int)
        
        acc = accuracy_score(data.y_val, y_pred)
        auc = roc_auc_score(data.y_val, val_proba)
        f1 = f1_score(data.y_val, y_pred)
        prec = precision_score(data.y_val, y_pred)
        rec = recall_score(data.y_val, y_pred)
        
        print(f"Accuracy : {acc:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print("-" * 80)
        print("Classification Report:")
        print(classification_report(data.y_val, y_pred, digits=4))
        print("=" * 80)
        
        # 4. 📝 LOGGING TO MLFLOW (ไฮไลท์ของการทำ MLOps)
        # Log ค่า Hyperparameters ตัวที่ชนะ
        mlflow.log_params(best_params)
        
        # Log ค่าผลการประเมิน
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("training_time", total_training_time)
        
        # Log ตัวโมเดล (บันทึกเป็นไฟล์ Pipeline พร้อมใช้)
        # สิ่งนี้จะเซฟโมเดลเก็บไว้ในโฟลเดอร์ mlruns อัตโนมัติ ไม่ต้องทำ joblib เอง
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="random_forest_pipeline",
            input_example=data.X_val_raw.iloc[[0]] # ยกตัวอย่างข้อมูลเข้าไปด้วย
        )
        
        print("🎉 Successfully logged parameters, metrics, and model to MLflow!")
        print("👉 Run 'mlflow ui' in your terminal to view the dashboard.")


if __name__ == "__main__":
    main()