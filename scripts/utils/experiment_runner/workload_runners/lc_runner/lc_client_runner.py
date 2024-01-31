from collections.abc import Callable, Iterable, Mapping
import os
import math
import time 
import subprocess
from typing import Any
import psutil
import threading
import math 
import numpy as np

# from lc_controller import LCController
# from lc_server_runner import LoadMonitor

MODEL_NAME = "resnet152"
CLIENT_CMD = f"/workloads/Inference-server/build/bin/client -m {MODEL_NAME} -c 1 -e trace -f "

class LCClientRunner:
    running = False
    trace_file = ""
    MODEL_NAME = "resnet152"
    CLIENT_CMD = f"/workloads/Inference-server/build/bin/client -m {MODEL_NAME} -c 1 -e trace -f "
    def __init__(self, trace_file: str, gpus: int, max_rps_per_gpu : int, trace_unit_sec: int = 60, debug: bool = False) -> None:
        self.proc = None
        self.debug = debug
        self.max_rps_per_gpu = max_rps_per_gpu
        if trace_unit_sec < 0 or trace_unit_sec > 60:
            print(f"Client runner got {trace_unit_sec} as trace time unit. Using default (=60 sec)")
            trace_unit_sec = 60
        self.trace_unit_sec = trace_unit_sec
        self.trace_file = self.process_workload_trace(trace_file, max_rps_per_gpu, trace_unit_sec, gpus)
        
    def process_workload_trace(self, workload_trace: str, per_gpu_max_rps: float, trace_unit: int, num_gpus: int):
        f = open(workload_trace, 'r')
        data = f.readlines()
        load_lst = "duration,rps\n"
        print(f"Client runner, normalizing {workload_trace} trace data to the inference server load")

        for line in data:
            load_lst += f"{trace_unit},{int((float(line) / 100.0) * num_gpus * per_gpu_max_rps)}\n"
        if self.debug: # saves con
            tmp_trace_name = workload_trace[workload_trace.rfind("/")+1:]
        tmp_trace_name = tmp_trace_name + f"_converted_{num_gpus}GPUs_{per_gpu_max_rps}maxRPSPerGPU"
        f = open(tmp_trace_name, 'w')
        f.write(load_lst)
        f.close()
        return tmp_trace_name
    
    def run_client(self, server_ip: str = "localhost", cmd: str = ""): # runs client
        if len(cmd) == 0:
            cmd = (self.CLIENT_CMD+f"{self.trace_file}")
        cmd =+ f" -i {server_ip}" # remote server support
        # run client
        print(f"Running the LC client {cmd} ... ", flush=True, end="")
        p = subprocess.Popen(
            cmd.split(" "),
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out, err = p.communicate()
        # wait to finish
        print("Done! ", flush=True, end="")
        p.wait()
        # print(out.decode())
        # print("Client done! ", flush=True)
        # parse client output and return rps and power
        return self.parse_client_output(out.decode())
    
    def parse_client_output(self, output: str):

        def break_to_key_val(data: str, prefix: str = "p95th"):
            def extract_power(line_list):
                power = 0.0
                for line in split_by_new_line:
                    split_by_colon = line.split(":")
                    if len(split_by_colon)!= 3:
                        continue
                    # print(f" line:{line} split:{split_by_colon}")
                    power += float(split_by_colon[2])
                return power

            def extract_latency(line_list):
                for line in split_by_new_line:
                    split_by_colon = line.split(":")
                    if len(split_by_colon)!= 3:
                        continue
                    if prefix in split_by_colon[1]:
                        return float(split_by_colon[2])

            split_by_new_line = data.split("\n")
            
            # print(split_by_new_line, flush="True")
            if "end2end" in split_by_new_line[0]:
                return extract_latency(split_by_new_line[1:])
            elif "power" in split_by_new_line[0]:
                return extract_power(split_by_new_line[1:])
        latency_str = output[output.find("end2end"): output.find("inference")]
        percentile_latency = dict()
        percentile_latency['min'] = break_to_key_val(latency_str, prefix='min')
        percentile_latency['max'] = break_to_key_val(latency_str, prefix='max')
        percentile_latency['mean'] = break_to_key_val(latency_str, prefix='avg')
        percentile_latency['p25th'] = break_to_key_val(latency_str, prefix='p25th')
        percentile_latency['p50th'] = break_to_key_val(latency_str, prefix='p50th')
        percentile_latency['p75th'] = break_to_key_val(latency_str, prefix='p75th')
        percentile_latency['p90th'] = break_to_key_val(latency_str, prefix='p90th')
        percentile_latency['p95th'] = break_to_key_val(latency_str, prefix='p95th')
        percentile_latency['p99th'] = break_to_key_val(latency_str, prefix='p99th')
        
        # powers_str = output[output.find("powers"): output.find("request-resouce-utils")]
        # power = break_to_key_val(powers_str)

        # print(f"l: {latency_str} p: {powers_str} el: {percentile_latency} ep: {power}", flush=True)
        return percentile_latency

#wrapper for client runner
class LCClientRunnerWarpper:
    client = None
    warmup_loads_trace = "/tmp/warmup_client_trace"
    max_warmup_load_pct = 50
    def __init__(self, num_warmpup_load_steps: int, warmup_step_duration_sec: int, trace_file: str, gpus: int, max_rps_per_gpu: int, trace_unit_sec: int = 60, debug: bool = False):
        self.trace_file=trace_file
        self.max_rps_per_gpu=max_rps_per_gpu
        self.trace_unit_sec=trace_unit_sec
        self.debug=debug
        self.gpus=gpus
        self.num_warmpup_load_steps = num_warmpup_load_steps
        self.warmup_step_duration_sec = warmup_step_duration_sec
    def warmup(self, server_ip: str = "localhost"):
        # generate warmpup trace
        step = math.ceil(self.max_warmup_load_pct/self.num_warmpup_load_steps)
        loads = list()
        for i in range(self.max_warmup_load_pct, 1, -step):
            loads.append(i)
        loads.reverse()
        f = open(self.warmup_loads_trace, 'w')
        for i in loads:
            f.write(f"{i}\n")
        f.close()
        
        print(f"warmingup the lc server with: min load {loads[0]}, max load: {loads[-1]}, step {step} for {self.warmup_step_duration_sec} sec each.")
        client = LCClientRunner(
            trace_file=self.warmup_loads_trace,
            gpus=self.gpus,
            max_rps_per_gpu=self.max_rps_per_gpu,
            trace_unit_sec=self.warmup_step_duration_sec,
            debug=self.debug
        )
        results = client.run_client(server_ip=server_ip)
        return results
    def run(self, server_ip: str = "localhost"):
        client = LCClientRunner(
            trace_file=self.trace_file,
            gpus=self.gpus,
            max_rps_per_gpu=self.max_rps_per_gpu,
            trace_unit_sec=self.trace_unit_sec,
            debug=self.debug
        )
        return client.run_client(server_ip=server_ip)

def test_lc_client_runner():
    import sys
    import os
    TRACE_FILE = "/tmp/tmp_trace"
    f = open(TRACE_FILE, 'w')
    f.write("30\n")
    f.write("50\n")
    f.write("70\n")
    f.write("80\n")
    # f.write("70\n")
    # f.write("30\n")
    f.close()
    runner = LCClientRunner(TRACE_FILE, 2, 150, 60, True)
    print(runner.run_client(CLIENT_CMD+TRACE_FILE+"_normalized150"))

def test_lc_client_runner_wrapper():
    import sys
    import os
    import sys
    import os
    TRACE_FILE = "/tmp/tmp_trace"
    f = open(TRACE_FILE, 'w')
    f.write("30\n")
    f.write("50\n")
    f.write("70\n")
    f.write("80\n")
    f.close()
    print("warming up ... ", end="")
    runner = LCClientRunnerWarpper(
        num_warmpup_load_steps=3,
        warmup_step_duration_sec=10,
        trace_file=TRACE_FILE,
        gpus=2,
        max_rps_per_gpu=150,
        trace_unit_sec=60, 
        debug=True
    )
    print("done!")
    warmup_res = runner.warmup()
    print(f"warmup results: {warmup_res}")
    print("Running ... ", end="")

    run_res = runner.run()
    print("done!")
    print(f"Run results: {run_res}")
    



if __name__ == "__main__":
    test_lc_client_runner_wrapper()