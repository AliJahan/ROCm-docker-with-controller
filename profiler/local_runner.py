
import sys
import threading
import subprocess
import multiprocessing
import time
# import rocm-smi lib
sys.path.append('/opt/rocm//libexec/rocm_smi/')
from rsmiBindings import *
import zmq
import subprocess
import psutil
import time
import threading
import copy
import os

BATCH_CMD = "python3 /transformers/examples/pytorch/language-modeling/run_clm_no_trainer.py --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --model_name_or_path gpt2 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --output_dir /tmp/test-clm --num_train_epochs "
LOG_PATH = "/workspace/profiler/gpt2_logs"

        
class BERunner(threading.Thread):
    procs = dict()
    ps_procs = dict()
    outputs = dict()
    lock = threading.Lock()
    dev = 0
    expr_ind = 0
    running = False
    log_file = None
    def __init__(self, run_cmd: str, expr_ind: int, debug = False):
        self.debug = debug
        self.run_cmd = run_cmd
        self.expr_ind = expr_ind
        # open log file to pass to popen as stdout and stderr pipes
        self.log_file = open(f"{LOG_PATH}/expr_{self.expr_ind}", 'w') 
        threading.Thread.__init__(self)

    def run(self):
        print(f"BERunner running on dev:{self.dev} log path: {LOG_PATH}/expr_{self.expr_ind}", flush=True)
        
        cmd = (self.run_cmd)
        my_envs = {
            **os.environ
        }
        print(f"CMD:\n{cmd}\nENV:{my_envs}", flush=True)
        # Run benchmark (cmd) in shell
        p = subprocess.Popen(
            cmd,
            env=my_envs,
            stdout=self.log_file,
            stderr=self.log_file,
            shell=True
        )
        
        ps = psutil.Process(pid=p.pid)
        with self.lock:
            self.procs[self.dev] = (p, ps)

    def get_throughput(self):        
        procs = dict()
        with self.lock:
            procs = copy.deepcopy(list(self.procs.keys()))
        print(f"reading {procs}", flush=True)
        for i in procs:
            p = procs[i][0]
            out, err = p.communicate()
            print(f"dev: {i} out:{out.decode()}", flush=True)

    def kill(self, proc):
        process = psutil.Process(proc.pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()

    def read_log(self):
        # reads data stored log during benchmark run
        self.log_file.flush()
        self.log_file.close()
        self.log_file = open(f"{LOG_PATH}/expr_{self.expr_ind}", 'r')
        data = self.log_file.read()
        self.log_file.close()
        return data

    def stop(self, dev):
        if self.debug:
            print(f"Terminating {dev}", flush=True)
        num_iter = 0
        throughput = 0
        with self.lock:
            if dev in self.procs:
                if self.procs[dev][1].is_running():
                    self.kill(self.procs[dev][0])
                    out = self.read_log()
                    s_cutoff = "INFO - __main__ -   Total optimization steps ="
                    out = out[out.find(s_cutoff)+len(s_cutoff)+10:]
                    if "\r\n" in out:
                        lines = out.split("\r\n")
                    elif "\n" in out:
                        lines = out.split("\n")
                    elif "\r" in out:
                        lines = out.split("\r")
                    else:
                        print(f"None of new line chars was found in output({out})", flush=True)
                        return "fail currupted output" 
                    lines = lines[2:]
                    data = ""
                    for line in lines:
                        if len(line) > 10:
                            line = line.strip()
                            line = line.split(",")[1]
                            unit = line[-5:].strip()
                            val = line[:-5].strip()
                            if len(val)>1:
                                p_v = val
                                if "it/s" in unit:
                                    val = 1.0/float(val)
                                elif "s/it" in unit:
                                    val = float(val)
                                else:
                                    print(f"unit: {unit} value: {val} could not extract from the report", flush=True)
                                    continue
                                throughput = float(throughput * num_iter + val) / float(num_iter+1)
                                num_iter += 1
                            if self.debug:
                                print(f"l({line}) u({unit}) v({p_v}) tp_sum({throughput}) tp_num({num_iter}) val({val})", flush=True)
                    
                    return str(throughput)
        return "fail terminate parsing end" # fail message as throughput to be sent to controller

class Profiler:
    def __init__(self, address: str = "localhost", port: str = "5951"):
        # setup zmq for communication
        context = zmq.Context()
        self.sock_ = context.socket(zmq.DEALER)
        
        try:
            self.sock_.bind("tcp://*:"+port)
            print("Connected to remote_profiler", flush=True)
        except:
            print(f"Could not connect to tcp://{address}:{port}", flush=True)

    def run(self):
        print(f"Profiler Running...", flush=True)
        running = False
        # Warmup for donwloading datasets
        tmp = BERunner(BATCH_CMD+"2", -1, True)
        tmp.start()
        time.sleep(30)
        res = tmp.stop(0)
        print(f"Warmup done with TP: {res}", flush=True)
        del tmp
        # comminucation protocol: the controller sends string messages containing commands. Commands:
        # - start: starts the benchmark
        # - stop: stops the benchmark and returns the thtoughput (i.e. secs/iteration) 
        expr_ind = 1 # keep track of number of experiments (logs are tagged with this value as a postfix)
        while True:
            print(f"/While", flush=True) # Mark start of while in logs
            # Wait for command
            req =  self.sock_.recv_string()
            print(f"Rcvd: {req}", flush=True)
            rep = ""
            if req == "START": # make sure it is not started before, then start
                if running == False:
                    self.runner = BERunner(BATCH_CMD+"5", expr_ind)
                    expr_ind += 1
                    self.runner.start()
                    running = True
                    rep = "success"
                else:
                    rep = "fail at start"
            elif req == "STOP":# make sure it is started before, then stop and return throughput
                if running == True:
                    running = False
                    rep = self.runner.stop(0)
                    self.runner = None
                else:
                    rep = "fail at stop"
            elif req == "READY": # ready is used for controller to make sure local_runner is up and ready
                rep = "success"
            elif req == "END": 
                break
            else:
                print(f"Requested command ({req}) does not exist!", flush=True)
                rep = "fail invalid cmd"
            # Send response to controller "fail*" or "sucess" or "average throughput value"
            self.sock_.send_string(rep)
            print(f"While/", flush=True) # Mark end of while
        print(f"Local runner stopped by controller request\n", flush=True)

def main():
    p = Profiler()
    p.run()


if __name__ == "__main__":
    main()