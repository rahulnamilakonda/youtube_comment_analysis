# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (for compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the backend requirements as requested
COPY backend/requirements.txt .

# Install dependencies into a separate prefix to keep final image small
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final Runtime Image
FROM python:3.11-slim

WORKDIR /app

# Install runtime-only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-installed python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy only the essentials as requested
COPY backend/ ./backend/
COPY models/ ./models/
COPY youtube_comment_analysis/config.py ./youtube_comment_analysis/config.py
COPY youtube_comment_analysis/__init__.py ./youtube_comment_analysis/__init__.py
COPY params.yaml .

# Set PYTHONPATH so the backend can import the core config
ENV PYTHONPATH=/app

# Expose production port
EXPOSE 8000

# Industrial Production Configuration:
# gunicorn: The process manager that keeps workers alive and robust.
# -w 4: Number of worker processes. Formula: (2 x $num_cores) + 1. 4 is ideal for 2-core EC2 instances.
# -k uvicorn.workers.UvicornWorker: Tells Gunicorn to use Uvicorn's high-speed ASGI engine for FastAPI.
# --bind 0.0.0.0:8000: Ensures the container listens to all external traffic (required for EC2/Docker).
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "backend.app.main:app", "--bind", "0.0.0.0:8000"]
