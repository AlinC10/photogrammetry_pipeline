# ==============================================================================
# Turntable Studio Dockerfile
# Headless Linux + Nvidia CUDA 12 + OpenMVS v2.3.0 + Python 3.10
# ==============================================================================

FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libglu1-mesa-dev \
    libxmu-dev \
    libxi-dev \
    libboost-all-dev \
    libopencv-dev \
    libcgal-dev \
    libeigen3-dev \
    ca-certificates \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Build VCGLib
WORKDIR /opt
RUN git clone https://github.com/cnr-isti-vclab/vcglib.git vcglib

# Build OpenMVS with CUDA support
WORKDIR /opt
RUN git clone https://github.com/cdcseacave/openMVS.git openMVS && \
    mkdir -p openMVS_build && \
    cd openMVS_build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DVCG_ROOT=/opt/vcglib \
          -DOpenMVS_USE_CUDA=ON \
          -DCMAKE_INSTALL_PREFIX=/opt/openmvs_bin \
          ../openMVS && \
    make -j$(nproc) && \
    make install

# ==============================================================================
# Final Runtime Stage
# ==============================================================================
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OPENMVS_BIN_DIR=/usr/local/bin/OpenMVS
ENV PATH="${OPENMVS_BIN_DIR}:${PATH}"

# Install Python 3.10 runtime & graphic libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libboost-iostreams1.74.0 \
    libboost-program-options1.74.0 \
    libboost-serialization1.74.0 \
    libboost-filesystem1.74.0 \
    libboost-system1.74.0 \
    libgmp10 \
    libmpfr6 \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy OpenMVS binaries from builder stage
COPY --from=builder /opt/openmvs_bin/bin /usr/local/bin/OpenMVS

# Setup workspace
WORKDIR /workspace/turntable_studio

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Default command
ENTRYPOINT ["python3", "main.py"]
CMD ["--input_dir", "/workspace/input_images", "--output_dir", "/workspace/output", "--rig_mode"]
