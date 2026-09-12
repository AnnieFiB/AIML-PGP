
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 user

WORKDIR /app

# Install dependencies first for better Docker caching
COPY deployment/requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy application files
COPY deployment/ /app/deployment/

# Copy model metadata generated during training
COPY models/model_output.txt /app/models/model_output.txt
COPY models/model_repo.txt /app/models/model_repo.txt

# Give ownership to non-root user
RUN chown -R user:user /app

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["streamlit", "run", "deployment/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.enableXsrfProtection=false"]
