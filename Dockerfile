# ---------------------------------------------------------------------------
# JourneyMind — single-service image: FastAPI serves the API and the built UI.
#
# Stage 1 builds the React bundle. Stage 2 is a slim CPU-only Python runtime
# with no PyTorch: the GNN is trained offline and served through a NumPy
# forward pass over exported weights, which keeps the image small and the
# memory footprint inside a small instance.
# ---------------------------------------------------------------------------
FROM node:20-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund
COPY frontend/ ./
# vite.config.js writes to ../backend/app/static. With WORKDIR /build that
# resolves to /backend/app/static, so no CLI override is needed — one source of
# truth for the output path, shared by the local build and this image.
RUN mkdir -p /backend/app/static && npm run build \
    && test -f /backend/app/static/index.html


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEMO_MODE=true \
    JM_DATA_DIR=/app/data \
    JM_MODELS_DIR=/app/models \
    JM_STATIC_DIR=/app/backend/app/static

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/
COPY data/ ./data/
COPY models/ ./models/
COPY --from=frontend /backend/app/static ./backend/app/static

# run as a non-root user
RUN useradd --create-home --uid 10001 journeymind && chown -R journeymind:journeymind /app
USER journeymind

WORKDIR /app/backend
EXPOSE 8000

# Reports what is actually loaded, not what was configured, so an image with a
# missing model bundle fails its healthcheck instead of serving fallbacks.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3     CMD python -c "import urllib.request,sys,json; b=json.load(urllib.request.urlopen('http://127.0.0.1:'+__import__('os').getenv('PORT','8000')+'/health',timeout=4)); sys.exit(0 if b.get('status')=='ok' else 1)"

# Render injects $PORT. Default to 8000 for plain `docker run`.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 65"]
