FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs/sessions logs/analytics qdrant_local_storage

# Expose ports
EXPOSE 8501 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LLM_PROVIDER=openai

# Default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
