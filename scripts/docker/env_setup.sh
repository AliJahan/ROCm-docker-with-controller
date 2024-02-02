#!/bin/bash

PROJECT_ROOT=$(dirname $(dirname "$PWD"))
echo $PROJECT_ROOT
LOGS_DIR=${PROJECT_ROOT}/docker_logs/
mkdir -p ${LOGS_DIR}