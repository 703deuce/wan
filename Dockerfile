# Use NVIDIA CUDA base image with Python (updated to supported version)
# Using CUDA 12.2.2 which is a valid and available tag
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

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

# Install PyTorch first (required for flash-attn build)
RUN pip3 install --no-cache-dir torch>=2.4.0 torchvision>=0.19.0 torchaudio>=2.4.0

# Install other dependencies (excluding flash-attn, transformers, diffusers, and peft - let Wan2.2 handle versions)
# Note: We don't install transformers/diffusers/peft here to avoid version conflicts with Wan2.2's requirements
RUN pip3 install --no-cache-dir \
    accelerate>=0.30.0 \
    safetensors>=0.4.0 \
    huggingface-hub>=0.20.0 \
    Pillow>=10.0.0 \
    opencv-python>=4.8.0 \
    imageio>=2.31.0 \
    imageio-ffmpeg>=0.4.9 \
    librosa>=0.10.0 \
    soundfile>=0.12.0 \
    numpy>=1.24.0 \
    requests>=2.31.0 \
    tqdm>=4.66.0 \
    runpod>=1.6.0 \
    einops>=0.7.0 \
    xformers>=0.0.23 \
    timm>=0.9.0 \
    omegaconf>=2.3.0 \
    pyyaml>=6.0.1

# Try to install flash-attn (optional, skip if it fails)
RUN pip3 install --no-cache-dir flash-attn>=2.5.0 || echo "flash-attn installation skipped (optional)"

# Clone Wan2.2 repository
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/wan2.2

# Install Wan2.2 and its dependencies
# Skip flash_attn since it's already installed (or optional) and causes build isolation issues
# Let Wan2.2's requirements handle transformers versions to ensure compatibility
RUN cd /app/wan2.2 && \
    grep -v "^flash_attn" requirements.txt > /tmp/wan2_requirements.txt && \
    pip3 install -r /tmp/wan2_requirements.txt && \
    if [ -f requirements_s2v.txt ]; then pip3 install -r requirements_s2v.txt; fi

# Install peft explicitly after Wan2.2 requirements to ensure compatibility
# diffusers 0.35.2 requires peft>=0.17.0, so install a compatible version
RUN pip3 install --no-cache-dir "peft>=0.17.0"

# Install huggingface-cli for model download
RUN pip3 install "huggingface_hub[cli]"

# Create model directory (model will be downloaded at runtime to avoid huge image size)
RUN mkdir -p /app/models

# Add Wan2.2 to Python path (fix undefined variable warning)
ENV PYTHONPATH=/app/wan2.2

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

