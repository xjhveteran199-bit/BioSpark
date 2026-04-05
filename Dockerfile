FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY training/ ./training/
COPY sample_data/ ./sample_data/

# Create uploads directory
RUN mkdir -p uploads

# Expose port (Railway sets $PORT dynamically)
EXPOSE 8000

# Start server — use $PORT for Railway compatibility, fallback to 8000
CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
