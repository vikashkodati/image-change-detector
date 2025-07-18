# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and git
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install uv

# Copy dependency files first for better Docker layer caching
COPY backend/pyproject.toml backend/uv.lock ./backend/

# Change to backend directory for UV operations
WORKDIR /app/backend

# Install Python dependencies including AI packages
RUN uv sync --frozen

# Force NumPy 1.x installation to fix compatibility issues
RUN echo "🔧 Forcing NumPy 1.x compatibility..." && \
    uv pip install "numpy>=1.24.0,<2.0.0" --force-reinstall

# Install CLIP with multiple fallback strategies
RUN echo "📥 Installing CLIP..." && \
    uv pip install "git+https://github.com/openai/CLIP.git" || \
    (echo "⚠️  Git CLIP install failed, trying pip fallback..." && \
     uv pip install clip-by-openai || \
     echo "❌ All CLIP installation methods failed - will run with OpenCV-only mode")

# Verify installations with NumPy compatibility check
RUN echo "🔍 Verifying installations..." && \
    uv run python -c "import numpy; print(f'✅ numpy: {numpy.__version__}'); assert numpy.__version__.startswith('1.'), f'NumPy 2.x detected: {numpy.__version__}'" && \
    uv run python -c "import torch; print(f'✅ torch: {torch.__version__}')" && \
    uv run python -c "import torchvision; print(f'✅ torchvision: {torchvision.__version__}')" && \
    (uv run python -c "import clip; print(f'✅ CLIP: {clip.available_models()}'); print('✅ CLIP initialization successful')" || \
     echo "⚠️  CLIP verification failed but torch is available")

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