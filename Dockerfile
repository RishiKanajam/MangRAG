FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into the project venv
RUN uv sync --frozen --no-dev

# Copy application code
COPY mangrag/ mangrag/
COPY api.py app.py main.py ./

EXPOSE 8000 8501

# Default: run the FastAPI server
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
