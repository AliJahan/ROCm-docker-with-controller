# ROCm-docker-with-controller
ROCM Docker with Controller in ROCR 


# How to run
Build an image docker with modified ROCR stack, Pytorch 1.13.0 compiled with ROCR stack and ready to use for profiling.
```bash
#pull the repo & checkout gpt2 profiling branch. from this repos root dir:
cd profiler;
profile.sh # builds the image witheverything needed, then runs the profiler for gpt2 training 
```
