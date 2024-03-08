import math
import time
import subprocess

MODEL_NAME = "resnet152"
CLIENT_CMD = f"/workloads/Inference-server/build/bin/client -m {MODEL_NAME} -c 1 -e trace -f "

class LCClientRunner:
    running = False
    trace_file = ""
    process = None
    MODEL_NAME = "resnet152"
    CLIENT_CMD = f"/workloads/Inference-server/build/bin/client -m {MODEL_NAME} -c 1 -e trace -f "
    def __init__(self, trace_file: str, gpus: int, max_rps_per_gpu : int, trace_unit_sec: int = 60, debug: bool = False) -> None:
        self.proc = None
        self.debug = debug
        self.max_rps_per_gpu = max_rps_per_gpu
        if trace_unit_sec < 0 or trace_unit_sec > 60:
            print(f"\t-[Client runner] Got {trace_unit_sec} as trace time unit. Using default (=60 sec)")
            trace_unit_sec = 60
        self.trace_unit_sec = trace_unit_sec
        self.wait_for_client_sec = max(5, int(0.1*self.trace_unit_sec)) # 10% more sleep time for client
        self.trace_file = self.process_workload_trace(trace_file, max_rps_per_gpu, trace_unit_sec, gpus)
        
    def process_workload_trace(self, workload_trace: str, per_gpu_max_rps: float, trace_unit: int, num_gpus: int):
        f = open(workload_trace, 'r')
        data = f.readlines()
        load_lst = "duration,rps\n"
        print(f"\t-[Client runner] Converting {workload_trace} trace data (load percent) to the inference server load (rps)")
        
        for line in data:
            load_lst += f"{trace_unit},{int((float(line) / 100.0) * num_gpus * per_gpu_max_rps)}\n"
            self.wait_for_client_sec += self.trace_unit_sec
        # Save converted load(%) to RPS
        tmp_trace_name = workload_trace + f"_converted_{num_gpus}GPUs_{per_gpu_max_rps}maxRPSPerGPU"
        f = open(tmp_trace_name, 'w')
        f.write(load_lst)
        f.close()
        return tmp_trace_name
    
    def run_client(self, server_ip: str = "localhost", cmd: str = "", blocking = True): # runs client
        if len(cmd) == 0:
            cmd = (self.CLIENT_CMD+f"{self.trace_file}")
        cmd += f" -i {server_ip}" # remote server support
        # run client
        print(f"\t-[Client runner] Running the LC client {cmd} ... ", flush=True, end="")
        self.process = subprocess.Popen(
            cmd.split(" "),
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # wait to finish
        print("Done! ", flush=True)
        if blocking:
            print(f"\t-[Client runner] sleeping for {self.wait_for_client_sec} sec.", flush=True)
        
            time.sleep(self.wait_for_client_sec)
            print(f"\t-[Client runner] checking if client process is done ... ", flush=True, end="")
            if self.process.poll() is None:
                print(f"No (retrying in 60 sec) ... ", flush=True, end="")
                time.sleep(self.trace_unit_sec)
            print(f"Done!", flush=True)
            
            # check for client
            if self.process.poll() is None:
                print(f"\t-[Client runner] ERROR client process not responsding!", flush=True)
                self.process.kill()
                return None
            
            self.process.wait()
            out, err = self.process.communicate()
            # print(out.decode())
            # print("Client done! ", flush=True)
            # parse client output and return rps and power
            return self.parse_client_output(out.decode())
        
    def get_lc_results(self):
        if self.process is None:
            return None
        sleep_sec = 180
        if self.process.poll() is None:
            print(f"\t-[Client runner] sleeping for {sleep_sec} sec.", flush=True)
        
            time.sleep(sleep_sec)
            print(f"\t-[Client runner] checking if client process is done ... ", flush=True, end="")
            if self.process.poll() is None:
                print(f"No (retrying in {sleep_sec} sec) ... ", flush=True, end="")
                time.sleep(sleep_sec)
            print(f"Done!", flush=True)
        
            # check for client
            if self.process.poll() is None:
                print(f"\t-[Client runner] ERROR client process not responsding!", flush=True)
                self.process.kill()
                return None
        
        self.process.wait()
        out, err = self.process.communicate()
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
        if "Client did not recieve any response" in output:
            return None
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
    max_warmup_load_pct = 20
    def __init__(
            self, 
            num_warmpup_load_steps: int, 
            warmup_step_duration_sec: int, 
            trace_file: str, 
            gpus: int, 
            max_rps_per_gpu: int, 
            trace_unit_sec: int = 60, 
            debug: bool = False
        ):
        self.trace_unit_sec = trace_unit_sec
        self.max_rps_per_gpu = max_rps_per_gpu
        self.num_warmpup_load_steps = num_warmpup_load_steps
        self.trace_file=trace_file
        self.gpus = gpus
        self.warmup_step_duration_sec = warmup_step_duration_sec
        self.debug = debug
        
        if len(self.trace_file) == 0:
            print(f"\t-[Client runner] No trace provided, using warmup trace {self.warmup_loads_trace}")
            self.generate_warmpup_trace()
            self.trace_file = self.warmup_loads_trace
        
    def generate_warmpup_trace(self):
        step = math.ceil(self.max_warmup_load_pct/self.num_warmpup_load_steps)
        
        loads = list()
        print("\t-[Client runner] warmup trace pct:", end="", flush=True)
        for i in range(self.max_warmup_load_pct, 1, -step):
            loads.append(i)
            print(f"{i},", end="", flush=True)
        print("", flush=True)
        loads.reverse()
        f = open(self.warmup_loads_trace, 'w')
        for i in loads:
            f.write(f"{i}\n")
        f.close()
        return loads, step

    def warmup(self, server_ip: str = "localhost"):
        # generate warmpup trace
        loads, step = self.generate_warmpup_trace()
        print(f"\t-[Client runner] Warmingup the lc server with: min load {loads[0]}, max load: {loads[-1]}, step {step} for {self.warmup_step_duration_sec} sec each.")
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
        file_name = self.trace_file.split('/')[-1]

        # Sometimes client fails, we try 3 times
        num_tries = 3
        res = None
        tries = 0
        while res == None and tries < num_tries:
            print(f"\t-[Client runner] Running the LC client: try:{tries} gpus:{self.gpus} trace_file:{file_name} trace_unit:{self.trace_unit_sec} sec each load.")
            res = client.run_client(server_ip=server_ip)
        return res

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