
FROM python:3.11-slim

# Set container working directory
WORKDIR /app

# Install dependencies first for Docker caching
COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 user

# Copy deployment files
COPY --chown=user:user . /app

# Run container as non-root user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Hugging Face Docker Spaces use port 7860
EXPOSE 7860

CMD [
    "streamlit",
    "run",
    "streamlit_app.py",
    "--server.address=0.0.0.0",
    "--server.port=7860",
    "--server.enableXsrfProtection=false"
]
