FROM python:3.11-slim

# libportaudio2 is only needed at runtime because requirements.txt still
# includes sounddevice (used by the standalone script.py CLI agent).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8420
EXPOSE 8420

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# gevent worker so the /ws/eva WebSocket endpoint (flask-sock) works
CMD ["sh", "-c", "gunicorn -k gevent -w 1 -b 0.0.0.0:${PORT} --timeout 120 app:app"]