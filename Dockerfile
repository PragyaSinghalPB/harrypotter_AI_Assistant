# Container image for Docker-based hosts (Render, Railway, Fly.io, a VPS).
# Vercel does not use this file; it deploys via vercel.json + api/index.py.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn==23.0.0

COPY . .

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# One worker keeps the in-memory agent/chat state consistent.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 120 starty:app"]
