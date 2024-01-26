#!/bin/bash

echo "REMOVING/STOPPING devcontainer-$1-1"
docker rm -f devcontainer-$1-1 