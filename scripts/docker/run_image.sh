#!/bin/bash
. ./env_setup.sh

cd ${PROJECT_ROOT}/.devcontainer/ && docker compose up -d $1 &> ${LOGS_DIR}/$(date -I'minutes')$1_run.log && cd ${PROJECT_ROOT}/scripts/docker