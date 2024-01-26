#!/bin/bash

PROJECT_ROOT=$(dirname $(dirname "$PWD"))
LOGS_DIR=${PROJECT_ROOT}/logs/
mkdir -p ${LOGS_DIR}

# GPU workloads need to be cloned since they are provate repos and need ssh keys to be cloned. 
if [ $1 = "gpu-workloads-remote-rocm" ]; then
echo "Cloning workloads into /workloads/ ..." && bash ${PROJECT_ROOT}/workloads/clone_workloads.sh ${PROJECT_ROOT}/workloads/
fi
echo "Building Image: $1 ..." && cd ${PROJECT_ROOT}/.devcontainer/ && docker compose build $1  --no-cache  --progress plain &> ${LOGS_DIR}/$(date -I'minutes')$1_build.log && cd ${PROJECT_ROOT}/scripts/docker && printf "\r\rDONE\n"
