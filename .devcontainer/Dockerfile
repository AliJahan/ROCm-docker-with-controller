FROM ubuntu:20.04
LABEL maintainer=peng.sun@amd.com

# Register the ROCM package repository, and install rocm-dev package
ARG ROCM_VERSION=5.2
ARG AMDGPU_VERSION=22.20
ARG CONTROL_PORT="9090"
ARG BUILD_MODE="Debug"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl libnuma-dev gnupg \
  && curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
  && printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION/ ubuntu main" | tee /etc/apt/sources.list.d/rocm.list \
  && printf "deb [arch=amd64] https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/ubuntu focal main" | tee /etc/apt/sources.list.d/amdgpu.list \
  && apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  sudo \
  git \
  cmake \
  libelf1 \
  kmod \
  file \
  libncurses-dev \
  libzmq3-dev \ 
  libudev-dev \
  python3 \
  python3-pip \
  rocm-dev \
  rocm-libs \
  build-essential && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
RUN groupadd -g 109 render

# Build and install ROCR with resource-control through SHM
RUN echo "Building ROCR with CUMasking through SHM" \
  && cd /tmp \
  && git clone https://github.com/AliJahan/ROCR-Runtime.git \
  && cd ROCR-Runtime \
  && git checkout rocm-5.2.0-resource-control-shm \
  && bash ./build-install-rocr.sh \
  && cd /tmp \
  && rm -rf ROCR-Runtime;
  # && mv /opt/rocm/lib/libhsa-runtime64.so.1.5.0 /opt/rocm/lib/libhsa-runtime64.so.1.5.50200;

ENV ROCM_PATH=/opt/rocm/

# Build and install controller
COPY ../controller /tmp/controller

RUN cd /tmp/controller \
  && mkdir build \
  && cd build \
  && cmake -DCMAKE_BUILD_TYPE="${BUILD_MODE}" -DCONTROL_PORT="${CONTROL_PORT}" .. \
  && make -j \
  && cd ../scripts/ \
  && bash setup_service.sh; 

EXPOSE "${CONTROL_PORT}"

# Start controller service () then keep container running for debug porposes
ENTRYPOINT /tmp/controller/scripts/start_service.sh && /tmp/controller/scripts/debug_wait.sh