FROM tmp.hocbgqj0dg:latest
LABEL maintainer=peng.sun@amd.com

# Register the ROCM package repository, and install rocm-dev package
ARG ROCM_VERSION=5.2
ARG AMDGPU_VERSION=22.20
ARG CONTROL_PORT="9090"
ARG BUILD_MODE="Debug"


USER root

# First 4 are needed for ROCR, libssl-dev for pytorch cmake, and the rest are controller's and dev tool's
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates \
  libnuma-dev \
  gnupg \
  libudev-dev \
  sudo \
  git \
  vim \
  cmake \
  libssl-dev \
  software-properties-common\
  libncurses-dev \
  libzmq3-dev \ 
  python3 \
  python3-dev \
  python3-pip \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
RUN groupadd -g 109 render

# Build and install ROCR with resource-control through SHM
RUN echo "Building ROCR with CUMasking through SHM" \
  && cd /tmp \
  && git clone https://github.com/AliJahan/ROCR-Runtime.git \
  && cd ROCR-Runtime \
  && git checkout 43a40d028fac6ff11cb1c92bf472444b33a85c01 \
  && bash ./build-install-rocr.sh \
  && cd /tmp \
  && rm -rf ROCR-Runtime\
  && rm /opt/rocm/lib/libhsa-runtime64.so.1.5.50200 \
  && ln -s /opt/rocm/lib/libhsa-runtime64.so.1.5.0 /opt/rocm/lib/libhsa-runtime64.so.1.5.50200;

# Prep for pytorch build
ENV ROCM_PATH=/opt/rocm/
ENV PYTORCH_ROCM_ARCH gfx906
ENV HIP_PLATFORM amd
ENV PATH /opt/cache/bin:/opt/rocm/llvm/bin:/opt/rocm/opencl/bin:/opt/rocm/hip/bin:/opt/rocm/hcc/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/bin
RUN pip3 install astunparse pyyaml numpy==1.21.2 typing-extensions mkl==2021.4.0 mkl-include==2021.4.0 setuptools cmake cffi typing_extensions future six requests dataclasses pillow

# Compile pytorch from source (this is necessary since Pytorch wheels come with pre-built rocm binaries which does not work with our modified ROCR)
RUN cd /tmp \
  && git clone --jobs `nproc` https://github.com/pytorch/pytorch.git \
  && cd pytorch \
  && git checkout v1.13.0 \
  && git submodule update --init --recursive --jobs `nproc`\
  && pip3 install torchvision==0.14.0 --no-deps \
  && python3 tools/amd_build/build_amd.py \
  && USE_ROCM=1 MAX_JOBS=`nproc` USER_NUMPY=1 python3 setup.py install \
  && USE_ROCM=1 MAX_JOBS=`nproc` USER_NUMPY=1 python3 setup.py install develop

# Build and install controller
COPY ../controller /tmp/controller
ENV PATH /opt/cache/bin:/opt/rocm/llvm/bin:/opt/rocm/opencl/bin:/opt/rocm/hip/bin:/opt/rocm/hcc/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/bin:/usr/sbin
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
# ENTRYPOINT /tmp/controller/scripts/debug_wait.sh