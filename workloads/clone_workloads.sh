#!/bin/bash

printf %s "-Cloning Inference-Server (Latency Critical)" && git clone --quiet git@github.com:AliJahan/Inference-server.git $1/Inference-server && printf "\n"
printf %s "-Cloning miniMDock (Best Effort)" && git clone --quiet git@github.com:AliJahan/miniMDock.git $1/miniMDock && printf "\n"
