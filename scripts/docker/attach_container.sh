#!/bin/bash

echo "Attaching to devcontainer-$1-1"
docker exec -it devcontainer-$1-1 bash