#!/usr/bin/env bash
# exit when any command fails
set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'if [ $? -ne 0 ]; then echo "\"${last_command}\" command failed with exit code $?."; fi;' EXIT;

pip3 install pyzmq && pip install --upgrade pyzmq
pip3 install psutil
git clone https://github.com/AliJahan/miniMDock.git && cd miniMDock && ./compile_hip_coe.sh && cd /workspace/profiler/ && python3 /workspace/profiler/local_runner.py &> local_runner.log &

# keeps the container up and running
tail -F /dev/null 