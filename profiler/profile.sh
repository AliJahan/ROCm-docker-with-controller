#!/usr/bin/env bash
# exit when any command fails
set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'if [ $? -ne 0 ]; then echo "\"${last_command}\" command failed with exit code $?."; fi;' EXIT;

SETUP_WAIT=180

function wait_file_changed {
    tail -fn0 "$1" | head -n1
}

cd ../.devcontainer/ && ./build_image.sh && ./run_image.sh

printf "Waiting for container to finish setting up local_runner..." > ../profiler/LOG && rm -rf ../profiler/local_runner.log && touch ../profiler/local_runner.log && wait_file_changed ../profiler/local_runner.log # Give container time to set up and start local_runner.py

echo "Done! Running remote_profiler now ... " >> ../profiler/LOG && cd ../profiler && ./build_master.sh && python3 remote_runner.py >> LOG && python3 plot.py