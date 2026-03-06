# ใช้ Python 3.10 เป็นฐาน
FROM python:3.10-slim

# ตั้งค่าโฟลเดอร์ทำงานใน Docker
WORKDIR /app

# ก๊อปปี้ไฟล์ requirements.txt เข้าไปและติดตั้ง
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ก๊อปปี้โค้ดทั้งหมด (รวมถึงโฟลเดอร์ mlruns ที่มีโมเดลอยู่) เข้าไปใน Docker
COPY . .

# เปิด Port 8000 สำหรับ API และ 8501 สำหรับ Streamlit
EXPOSE 8000 8501