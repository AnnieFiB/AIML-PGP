
FROM python:3.10-slim

WORKDIR /app

COPY deployment_files/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY deployment_files/ ./deployment_files/

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "deployment_files.app:app"]
