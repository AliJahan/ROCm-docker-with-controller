
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
BATCH_CMD = "/miniMDock/bin/autodock_hip_gpu_64wi -lfile /miniMDock/input/7cpa/7cpa_ligand.pdbqt -nrun 1000 -fldfile /miniMDock/input/7cpa/7cpa_protein.maps.fld"


        
class BERunner(threading.Thread):
    procs = dict()
    outputs = dict()
    lock = threading.Lock()
    dev = 0
    expr_ind = 0
    running = False
    def __init__(self, run_cmd: str, expr_ind: int, debug = False):
        self.debug = debug
        self.run_cmd = run_cmd
        self.expr_ind = expr_ind
        threading.Thread.__init__(self)
    def start_prof(sefl, dev):
        self.dev = dev
        self.start()

    def run(self):
        print(f"BERunner running on dev:{self.dev}", flush=True)
        my_envs = {
            **os.environ,
            "CUMASKING_CONTROLLER_LOG": f'expr_{self.expr_ind}'
        }
        p = subprocess.Popen(
            (self.run_cmd + f" -devnum {self.dev}").split(" "),
            env=my_envs,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE

        )
        
        ps = psutil.Process(pid=p.pid)
        with self.lock:
            self.procs[self.dev] = (p, ps)
        
    
    def run_and_suspend(self, dev):
        self.run(dev)
        time.sleep(1)
        self.suspend(dev)
    
    
    def get_throughput(self):
        
        procs = dict()
        with self.lock:
            procs = copy.deepcopy(list(self.procs.keys()))
        print(f"reading {procs}", flush=True)
        for i in procs:
            p = procs[i][0]
            out, err = p.communicate()
            print(f"dev: {i} out:{out.decode()}", flush=True)
            
    
    def suspend(self, dev):
        if self.debug:
            print(f"Suspending {dev}", flush=True)
        if dev in self.procs:
          self.procs[dev][1].suspend()

    def resume(self, dev):
        if self.debug:
            print(f"Resuming {dev}", flush=True)
        if dev in self.procs:
          self.procs[dev][1].resume()
    
    def terminate(self, dev):
        if self.debug:
            print(f"Terminating {dev}", flush=True)
        if dev in self.procs:
            if self.procs[dev][1].is_running():
                self.procs[dev][1].kill()
                out, err = self.procs[dev][0].communicate()
                out = out.decode()
                # print(len(out), flush=True)
                cutoff = "Local-search chosen method is: Solis-Wets (sw)\n0\n"
                out = out[out.find(cutoff)+len(cutoff):]
                return out
            
        return 0
        #   del self.procs[dev]

    # def __del__(self):
        # if self.debug:
        #     print("Terminating...", end="", flush=True)
        # devs = list(self.procs.keys())
        
        # for dev in devs:
        #     print(self.procs[dev], flush=True)
        #     self.terminate(dev)
        #     if self.debug:
        #         print(f"{dev}", end=" ", flush=True)
        # if self.debug:
        #     print("Done!", flush=True)

class Profiler:
    def __init__(self, address: str = "localhost", port: str = "5951"):
        context = zmq.Context()
        self.sock_ = context.socket(zmq.DEALER)
        
        try:
            self.sock_.connect("tcp://"+address+":"+port)
            print("Connected to remote_profiler", flush=True)
        except:
            print(f"Could not connect to tcp://{address}:{port}", flush=True)

    def run(self):
        print(f"Profiler waiting...", flush=True)
        running = False
        expr_ind = 1
        while True:
            print(f"/While", flush=True)
            req =  self.sock_.recv_string()
            print(f"Rcvd: {req}", flush=True)
            # req =  self.sock_.recv_string()
            # print(f"Rcvd: {req}", flush=True)
            rep = ""
            if req == "START":
                if running == False:
                    self.runner = BERunner(BATCH_CMD, expr_ind)
                    expr_ind += 1
                    self.runner.start()
                    running = True
                    rep = "success"
                else:
                    rep = "fail"
            elif req == "STOP":
                if running == True:
                    running = False
                    rep = self.runner.terminate(0)
                    self.runner = None
                else:
                    rep = "fail"
            else:
                print(f"Requested command ({req}) does not exist!", flush=True)
                rep = "fail2"
            self.sock_.send_string(rep)
            print(f"While/", flush=True)


def main():
    p = Profiler()
    p.run()


if __name__ == "__main__":
    main()