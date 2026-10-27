# ==============================================================================
# Turntable Studio Dockerfile
# Headless Linux + Nvidia CUDA 12 + OpenMVS v2.3.0 + mvs-texturing + Python 3.10
#
# Two stages: everything is compiled in the -devel image; the runtime image starts from the small
# nvidia/cuda:*-base and receives only the OpenMVS apps the pipeline runs, texrecon and the shared
# libraries ldd says they need (docker/runtime_deps.sh), instead of the -dev packages (OpenCV, Ceres,
# SuiteSparse...) the previous image installed. The build fails if any library is missing.
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
    libglew-dev \
    libglfw3-dev \
    libboost-all-dev \
    libopencv-dev \
    libcgal-dev \
    libeigen3-dev \
    libtbb-dev \
    libsuitesparse-dev \
    libceres-dev \
    nlohmann-json3-dev \
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

# 4. OpenMVS with CUDA support (pinned strictly to stable release v2.3.0)
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so || true && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1 || true && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 || true && \
    git clone --branch v2.3.0 --depth 1 https://github.com/cdcseacave/openMVS.git openMVS && \
    mkdir -p openMVS_build && \
    cd openMVS_build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DVCG_ROOT=/opt/vcglib \
          -DOpenMVS_USE_CUDA=ON \
          -DCUDA_CUDA_LIBRARY=/usr/local/cuda/lib64/stubs/libcuda.so \
          -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
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

# 6. Runtime payload: only the OpenMVS apps the pipeline runs (no Viewer), texrecon, and their libraries
COPY docker/runtime_deps.sh /opt/runtime_deps.sh
RUN mkdir -p /opt/payload/bin && \
    for app in InterfaceCOLMAP DensifyPointCloud ReconstructMesh RefineMesh TextureMesh; do \
        cp "$(find /opt/openmvs_bin/bin -type f -name "$app" | head -n 1)" /opt/payload/bin/; \
    done && \
    cp /opt/mvs-texturing/build/apps/texrecon/texrecon /opt/payload/bin/ && \
    sh /opt/runtime_deps.sh /opt/payload/lib /opt/payload/bin/*
# Debug symbols are not needed at run time (a failed strip only costs size, so it is not fatal)
RUN strip --strip-unneeded /opt/payload/bin/* /opt/payload/lib/*.so* 2>/dev/null || true

# ==============================================================================
# Final Runtime Stage
# ==============================================================================
FROM nvidia/cuda:12.2.2-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OPENMVS_BIN_DIR=/usr/local/bin/OpenMVS
ENV TEXRECON_BIN=/usr/local/bin/OpenMVS/texrecon
ENV PATH="${OPENMVS_BIN_DIR}:${PATH}"
# Undistorted images and OpenMVS files: fast container disk, not the bind-mounted output folder
ENV TURNTABLE_WORK_DIR=/tmp/turntable_work

# Python runtime only (no python3-dev, no -dev libraries). libgl1/libglib2.0-0/libgomp1: PyMeshLab and OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/payload/bin /usr/local/bin/OpenMVS
COPY --from=builder /opt/payload/lib /usr/local/lib/turntable
RUN echo /usr/local/lib/turntable > /etc/ld.so.conf.d/turntable.conf && ldconfig && \
    missing="$(for b in /usr/local/bin/OpenMVS/*; do ldd "$b" | grep 'not found' | grep -v -e libcuda -e libnvidia; done)"; \
    if [ -n "$missing" ]; then echo "Missing shared libraries:"; echo "$missing"; exit 1; fi

WORKDIR /workspace/turntable_studio

# Dependencies before the code: a code change only rebuilds the last small layers
COPY requirements-runtime.txt .
RUN pip3 install --no-cache-dir -r requirements-runtime.txt

# Code version recorded in stats.json (.git is not copied into the image). After the pip layer, so a new
# commit does not reinstall the Python packages:
#   docker build --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) ...
ARG GIT_COMMIT=unknown
ENV TURNTABLE_GIT_COMMIT=${GIT_COMMIT}

# Application source code (.dockerignore keeps out images, outputs, tests, docs, plans and tools)
COPY . .
RUN chmod +x docker_entrypoint.sh

# RunPod pod (JOB_ID set): runpod_job.py. Otherwise main.py with the given options.
ENTRYPOINT ["./docker_entrypoint.sh"]
CMD ["--input_dir", "/workspace/input_images", "--output_dir", "/workspace/output", "--rig_mode"]
