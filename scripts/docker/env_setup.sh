#!/bin/bash

PROJECT_ROOT=$(dirname $(dirname "$1"))
echo $PROJECT_ROOT
LOGS_DIR=${PROJECT_ROOT}/docker_logs/
mkdir -p ${LOGS_DIR}