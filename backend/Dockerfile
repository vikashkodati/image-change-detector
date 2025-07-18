# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install uv

# Copy dependency files first for better Docker layer caching
COPY backend/pyproject.toml backend/uv.lock ./backend/

# Change to backend directory for UV operations
WORKDIR /app/backend

# Install Python dependencies (basic + AI optional)
RUN uv sync --frozen && \
    uv sync --frozen --extra ai || echo "⚠️  AI dependencies failed to install - will run with basic features only"

# Copy source code
COPY backend/src/ ./src/

# Copy environment file if it exists (optional)
COPY backend/.env* ./

# Set Python path to include backend source
ENV PYTHONPATH=/app/backend/src

# Expose port
EXPOSE 8000

# Add health check with longer timeout for AI loading
HEALTHCHECK --interval=30s --timeout=60s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application using UV
CMD ["uv", "run", "python", "src/change_detector/server.py"] 