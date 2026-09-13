FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PREDICTION_API_URL=http://127.0.0.1:8000

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 127.0.0.1 --port 8000 & exec streamlit run streamlit.py --server.address=0.0.0.0 --server.port=7860 --server.headless=true"]
