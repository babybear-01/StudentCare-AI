# StudentCare-AI
StudentCare AI — A Production-Ready, Explainable Machine Learning System for Early Academic Risk Detection (SDG4)


## How to continue this project

### 1. Clone repository
git clone <repo-url>
cd StudentCare-AI

### 2. Checkout handoff branch
git checkout handoff-mlflow-docker

### 3. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

### 4. Install dependencies
pip install -r requirements.txt

### 5. Open MLflow UI
mlflow ui

### 6. Run backend API
python -m uvicorn src.api.app:app --reload

### 7. Run frontend
python -m streamlit run src/frontend/app.py