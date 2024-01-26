#!/bin/bash

PROJECT_ROOT=$(dirname $(dirname "$PWD"))
LOGS_DIR=${PROJECT_ROOT}/docker_logs/
mkdir -p ${LOGS_DIR}