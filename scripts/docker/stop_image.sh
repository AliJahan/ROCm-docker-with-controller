#!/bin/bash

. ./env_setup.sh

echo "REMOVING/STOPPING devcontainer-$1-1"
docker rm -f devcontainer-$1-1 &> ${LOGS_DIR}/$(date -I'minutes')$1_stop.log