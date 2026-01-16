# Jyotish AI Dockerfile
# Base: Python 3.11 slim for smaller image
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps (curl, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy dependency manifest
COPY requirements.txt ./

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Healthcheck (optional): verify streamlit responds
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: run Streamlit app
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
