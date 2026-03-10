FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data

# เปิดพอร์ตที่ใช้สำหรับ Streamlit
EXPOSE 8501

# กำหนดคำสั่งที่จะรันแอป Streamlit
CMD ["streamlit", "run", "src/frontend/app.py"]