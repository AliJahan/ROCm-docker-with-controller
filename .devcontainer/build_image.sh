#!/usr/bin/env bash
# exit when any command fails

IMAGE_NAME=gpu-controller

set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'if [ $? -ne 0 ]; then echo "\"${last_command}\" command failed with exit code $?."; fi;' EXIT;

sudo docker build .. -f Dockerfile -t $IMAGE_NAME