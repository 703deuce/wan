# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY handler.py /app/handler.py

# Create directories for model cache and temporary files
RUN mkdir -p /app/models /app/tmp

# Set environment variables for model paths
ENV MODEL_CACHE_DIR=/app/models
ENV TMP_DIR=/app/tmp
ENV HF_HOME=/app/models

# Expose port for RunPod
EXPOSE 8000

# Run the handler
CMD ["python", "handler.py"]

