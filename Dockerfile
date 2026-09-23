FROM python:3.11-slim

# libportaudio2 is only needed at runtime because requirements.txt still
# includes sounddevice (used by the standalone script.py CLI agent).
# supervisor runs both app.py (gunicorn) and agent.py (LiveKit worker)
# as separate processes inside this one container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 \
        curl \
        supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m livekit.agents download-files

COPY . .

COPY supervisord.conf /etc/supervisor/conf.d/eva.conf

ENV PORT=8420
EXPOSE 8420

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# supervisord now owns process startup - it launches both eva-web
# (gunicorn/app.py) and eva-agent (agent.py) and restarts either one
# independently if it crashes.
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/eva.conf", "-n"]