# LitSense model service (service/README.md, decisions D80). Build from the repo root:
#   docker build -f service/Dockerfile -t litsense-service .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    BUNDLE_DIR=/var/litsense \
    PORT=8000

# lightgbm needs the OpenMP runtime; without it the ranker silently falls back to the blend (D74).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin litsense \
    && mkdir -p /var/litsense \
    && chown litsense:litsense /var/litsense

WORKDIR /app
COPY requirements-core.txt requirements-service.txt ./
RUN pip install -r requirements-core.txt -r requirements-service.txt
COPY litsense_core ./litsense_core
COPY service ./service
# Precompiled bytecode: at 0.1 CPU every second of start-up counts.
RUN python -m compileall -q litsense_core service

USER litsense
EXPOSE 8000
# /ready is 200 only once the bundle is loaded; the start period covers download and load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/ready' % os.environ.get('PORT', '8000'), timeout=4)"]
CMD ["python", "-m", "service"]
