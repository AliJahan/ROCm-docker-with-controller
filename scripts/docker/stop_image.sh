#!/bin/bash
if [ -z "${PROJECT_ROOT}" ]; then
. ./env_setup.sh
fi 

echo "REMOVING/STOPPING devcontainer-$1-1"
docker rm -f devcontainer-$1-1 &> ${LOGS_DIR}/$(date -I'minutes')$1_stop.log