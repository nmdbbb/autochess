FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Ray
RUN pip3 install --no-cache-dir ray[default]

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints runs

# Expose Ray ports
EXPOSE 6379 8076 8265

# Command to run the application
CMD ["python3", "distributed/train_distributed.py"] 