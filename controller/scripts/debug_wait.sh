#!/usr/bin/env bash
# exit when any command fails
set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'if [ $? -ne 0 ]; then echo "\"${last_command}\" command failed with exit code $?."; fi;' EXIT;

# # local_runner requirements
# pip3 install pyzmq && pip install --upgrade pyzmq
# pip3 install psutil

# pip3 install  torchvision==0.14.0+rocm5.2 torchaudio==0.13.1 --no-deps --extra-index-url https://download.pytorch.org/whl/rocm5.2
# pip3 install huggingface-hub && pip3 install --upgrade huggingface-hub && pip3 install git+https://github.com/huggingface/transformers 
# pip3 install regex tqdm numpy matplotlib accelerate>=0.12.0 datasets>=1.8.0 sentencepiece!=0.1.92 protobuf evaluate scikit-learn && git clone --jobs `nproc` https://github.com/huggingface/transformers.git && cd transformers/examples/pytorch/language-modeling && python3 run_clm_no_trainer.py --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --model_name_or_path gpt2 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --output_dir /tmp/test-clm --max_train_steps 1 && mkdir -p /workspace/profiler/gpt2_logs && chmod ugo+rwx /workspace/profiler/gpt2_logs && cd /workspace/profiler/gpt2_logs  && python3 ../local_runner.py &> ../local_runner.log &

# keeps the container up and running
tail -F /dev/null 