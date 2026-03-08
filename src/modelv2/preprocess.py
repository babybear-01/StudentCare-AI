import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# 1. โหลดข้อมูล
input_path = Path(r"C:\Ject_MLOps\StudentCare-AI\data\processed\student-por-id.csv")
df = pd.read_csv(input_path)

# 2. เลือก Features ที่ต้องการ (รวมทั้งวิชาการและสภาพแวดล้อม)
features = [
    'G1', 'G2', 'G3', 'studytime', 'failures', 'absences',
    'Medu', 'Fedu', 'famrel', 'freetime', 'goout',
    'nursery', 'internet', 'romantic', 'Dalc', 'Walc','student_id'
]
df_clean = df[features].copy()

# 3. แปลงค่า Text ให้เป็นตัวเลข (Encoding)
le = LabelEncoder()
cat_cols = ['nursery', 'internet', 'romantic']
for col in cat_cols:
    df_clean[col] = le.fit_transform(df_clean[col])

# 4. บันทึกไฟล์ใหม่
data_save = Path(r"C:\Ject_MLOps\StudentCare-AI\data\processed\process_for_modelv2")

# สร้างโฟลเดอร์ ถ้ายังไม่มี
data_save.mkdir(parents=True, exist_ok=True)

# ต้องเป็น "ชื่อไฟล์.csv" ไม่ใช่แค่ชื่อโฟลเดอร์
output_filename = data_save / "porv2.csv"

df_clean.to_csv(output_filename, index=False)
print(f"บันทึกไฟล์สะอาดเรียบร้อยแล้วที่: {output_filename}")