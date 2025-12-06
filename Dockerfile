# NVIDIA CUDA runtime base so PyTorch can access GPU in-container
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Python 3.10 and essentials
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ensure `python` is available
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip

# Set Python env for reliable logs and fewer pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /dgm

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt /dgm/requirements.txt

# Install CUDA-enabled PyTorch first (uses cu121 wheels compatible with CUDA 12.x)
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Install the remaining Python dependencies
RUN python -m pip install --no-cache-dir -r /dgm/requirements.txt

# Now copy the rest of the repository
COPY . /dgm

# Keep the container running by default
CMD ["tail", "-f", "/dev/null"]