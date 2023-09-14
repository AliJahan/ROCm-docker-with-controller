#!/usr/bin/env bash
# exit when any command fails

set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'if [ $? -ne 0 ]; then echo "\"${last_command}\" command failed with exit code $?."; fi;' EXIT;

TAG=$(<running_tag)
sudo docker exec -it $TAG bash