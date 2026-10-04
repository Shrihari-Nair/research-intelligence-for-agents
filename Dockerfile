# API image: FastAPI + router + agents. The trained router weights (models/router) are mounted at runtime, not baked in.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf
# WeasyPrint (PDF reports) needs pango; curl is for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b fonts-dejavu-core curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN grep -vE '^(pytest|pytest-asyncio|ruff)$' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt
# bake the memory embedder so the first run needs no download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
COPY research_os research_os
COPY configs configs
COPY migrations migrations
COPY alembic.ini .
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=10 CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "research_os.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
