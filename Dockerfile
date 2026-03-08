FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV (headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Pre-download models at build time for faster cold starts
RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet512')" || true

EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --timeout 300 --workers 1 app:app"]
