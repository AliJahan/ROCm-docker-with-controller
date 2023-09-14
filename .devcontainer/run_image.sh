#!/usr/bin/env bash
# exit when any command fails

IMAGE_NAME=gpu-controller

set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'if [ $? -ne 0 ]; then echo "\"${last_command}\" command failed with exit code $?."; fi;' EXIT;

sudo docker run -d -it --privileged --device=/dev/kfd --device=/dev/dri --network host -v `echo $(cd ../ && pwd)`:/workspace --security-opt seccomp=unconfined --group-add video $IMAGE_NAME > running_tag
# sudo docker run -it --privileged --device=/dev/kfd --device=/dev/dri --network host -v `echo $(cd ../ && pwd)`:/workspace --security-opt seccomp=unconfined --group-add video $IMAGE_NAME 