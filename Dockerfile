FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Pre-download the CLIP model during build (not at runtime)
RUN python -c "from transformers import CLIPProcessor, CLIPModel; \
    CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14'); \
    CLIPModel.from_pretrained('openai/clip-vit-large-patch14')"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Expose port
EXPOSE 8000

# Run the API (Railway sets $PORT)
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
