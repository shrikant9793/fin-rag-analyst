# =============================================================================
# fin-rag-analyst — docker/Dockerfile
# Multi-stage build: builder (deps) → runtime (slim final image)
# Python 3.11-slim | ~1.2GB final image (embedding models included)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder: install all Python deps into a prefix
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some packages (unstructured, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpoppler-cpp-dev \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy only dependency files first (Docker cache optimisation)
COPY requirements.txt .

# Install all Python deps into /install prefix
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download embedding models so the container doesn't need internet at runtime
# This bakes the models into the image (~500 MB) — remove if you prefer lazy loading.
RUN pip install --no-cache-dir sentence-transformers fastembed && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')" && \
    python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm42-all-minilm-l6-v2-attentions')"

# ---------------------------------------------------------------------------
# Stage 2 — runtime: lean final image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="you@example.com"
LABEL description="Multi-Agent RAG Financial Market Research Analyst"
LABEL version="0.1.0"

WORKDIR /app

# Runtime system libraries (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpoppler-cpp-dev \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy pre-downloaded model cache from builder
COPY --from=builder /root/.cache /root/.cache

# Copy application source
COPY src/       /app/src/
COPY ui/        /app/ui/
COPY config/    /app/config/
COPY scripts/   /app/scripts/

# Create data directory for runtime artefacts (checkpoints, raw docs)
RUN mkdir -p /app/data/raw /app/data/processed

# Non-root user for security
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Healthcheck — Streamlit responds on /_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -sf http://localhost:8501/_stcore/health || exit 1

# Default command: launch Streamlit UI
CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]