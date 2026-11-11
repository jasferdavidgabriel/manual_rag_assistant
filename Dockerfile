# Manual RAG Assistant: web app + CLI in one image (CPU only).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

# Tesseract lets scanned manuals be OCR'd
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first: avoids pulling multi-GB GPU libraries
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so the first question isn't slow
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

# Run as a non-root user
RUN useradd --create-home appuser \
    && mkdir -p data/raw models reports/figures \
    && chown -R appuser /app
USER appuser

EXPOSE 7860 8000
CMD ["python", "app.py", "--host", "0.0.0.0"]
