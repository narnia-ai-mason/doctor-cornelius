# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy source code and README
COPY src ./src
COPY README.md ./

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy virtual environment and source code from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "doctor_cornelius.main:app", "--host", "0.0.0.0", "--port", "8000"]
