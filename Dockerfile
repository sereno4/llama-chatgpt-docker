FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake libopenblas-dev curl && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
COPY backend/ /app/backend/
COPY frontend/static/ /app/frontend/static/
RUN mkdir -p /app/models
EXPOSE 8001 8002
ENV PYTHONUNBUFFERED=1 MODEL_PATH=/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf N_THREADS=4 N_CTX=2048
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD curl -f http://localhost:8001/health || exit 1
CMD ["sh", "-c", "cd /app/backend && python server.py & cd /app/frontend && python -m http.server 8002 --directory static"]
