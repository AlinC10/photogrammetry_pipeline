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
    libtbb-dev \
    ca-certificates \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install modern CMake (>= 3.24), nanoflann and tinyxml2 required by OpenMVS & TinyEXIF
RUN wget -qO- "https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local && \
    apt-get update && apt-get install -y --no-install-recommends libnanoflann-dev libtinyxml2-dev && rm -rf /var/lib/apt/lists/*

# Build VCGLib, TinyEXIF, OpenMVS, and mvs-texturing
WORKDIR /opt

# 1. VCGLib
RUN git clone https://github.com/cnr-isti-vclab/vcglib.git vcglib

# 2. tinyxml2 (required by TinyEXIF with modern CMake package config)
RUN git clone --depth 1 https://github.com/leethomason/tinyxml2.git tinyxml2 && \
    mkdir -p tinyxml2_build && \
    cd tinyxml2_build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DBUILD_SHARED_LIBS=OFF \
          ../tinyxml2 && \
    make -j$(nproc) && \
    make install

# 3. TinyEXIF (metadata parser required by OpenMVS)
RUN git clone https://github.com/cdcseacave/TinyEXIF.git TinyEXIF && \
    mkdir -p TinyEXIF_build && \
    cd TinyEXIF_build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DCMAKE_PREFIX_PATH="/usr/local" \
          -DBUILD_SHARED_LIBS=OFF \
          ../TinyEXIF && \
    make -j$(nproc) && \
    make install

# 4. OpenMVS with CUDA support
RUN git clone https://github.com/cdcseacave/openMVS.git openMVS && \
    mkdir -p openMVS_build && \
    cd openMVS_build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DVCG_ROOT=/opt/vcglib \
          -DOpenMVS_USE_CUDA=ON \
          -DCMAKE_INSTALL_PREFIX=/opt/openmvs_bin \
          -DCMAKE_PREFIX_PATH="/usr/local" \
          ../openMVS && \
    make -j$(nproc) && \
    make install

# 5. mvs-texturing (texrecon from TU Darmstadt)
RUN git clone https://github.com/nmoehrle/mvs-texturing.git mvs-texturing && \
    mkdir -p mvs-texturing/build && \
    cd mvs-texturing/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-template-body -Wno-template-id-cdtor" .. && \
    make -j$(nproc)

# ==============================================================================
# Final Runtime Stage
# ==============================================================================
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OPENMVS_BIN_DIR=/usr/local/bin/OpenMVS
ENV TEXRECON_BIN=/usr/local/bin/texrecon
ENV PATH="${OPENMVS_BIN_DIR}:${PATH}"

# Install Python 3.10 runtime & graphic libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglu1-mesa \
    libglib2.0-0 \
    libgomp1 \
    libtbb2 \
    libpng16-16 \
    libjpeg-turbo8 \
    libtiff5 \
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

# Copy OpenMVS and texrecon binaries from builder stage
COPY --from=builder /opt/openmvs_bin/bin /usr/local/bin/OpenMVS
COPY --from=builder /opt/mvs-texturing/build/apps/texrecon/texrecon /usr/local/bin/texrecon

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
