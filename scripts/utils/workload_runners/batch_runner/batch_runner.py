import os
import zmq
import time
import psutil
import threading
import subprocess
import multiprocessing




class BatchRunner:
    run_cmd = "/workloads/miniMDock/bin/autodock_hip_gpu_64wi -lfile /workloads/miniMDock/input/7cpa/7cpa_ligand.pdbqt -nrun 1000 -fldfile /workloads/miniMDock/input/7cpa/7cpa_protein.maps.fld -devnum "
    procs = None
    lock = threading.Lock()
    throughput_reader_running = False
    throughput_reader_thread = None
    debug = False
    throughput_queue = multiprocessing.Queue()
    total_tp = (0.0, 0) # tp(iteration/sec), iterations
    logs_dir = "/workspace/logs"
    proc_log_file = None
    def __init__(self, debug = False):
        self.debug = debug
        # threading.Thread.__init__(self)

    def run(self, gpu: int):
        cmd = self.run_cmd + str(int(gpu)+1)
        # CUMASKING_CONTROLLER_LOG is set for cumasking controller (required for rocr to activate cumasking)
        env = {
            **os.environ,
            "CUMASKING_CONTROLLER_LOG": f"{self.logs_dir}/be_rocr"
        }

        # self.proc_log_file = open(f"{self.logs_dir}/be_run_log_g{int(gpu)+1}", 'w')
        p = subprocess.Popen(
            cmd.split(" "),
            env=env,
            # stdout=self.proc_log_file,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            # stderr=self.proc_log_file
            stderr=subprocess.PIPE
        )
        
        ps = psutil.Process(pid=p.pid)
        with self.lock:
            self.procs = (p, ps)
            self.throughput_reader_thread = threading.Thread(target=self.throughput_reader)
            self.throughput_reader_thread.start()
            
    
    def run_and_suspend(self, gpu):
        self.run(gpu)
        time.sleep(1)
        self.suspend()
    
    def calculate_average(self, output_lines):
        if self.debug:
            print(output_lines)
        if len(output_lines) == 0:
            return 0.0
        sum = 0
        for i in output_lines:
            sum += float(i)
        avg = float(sum)/len(output_lines)
        num_samples = 0
        with self.lock:
            self.total_tp = ((self.total_tp[0]*self.total_tp[1]+sum)/(self.total_tp[1]+len(output_lines)), self.total_tp[1]+len(output_lines))
            num_samples = self.total_tp[1]
        return avg, num_samples

    def throughput_reader(self):
        with self.lock:
            self.throughput_reader_running = True
        include_line = False
        while True:
            with self.lock:
                if self.throughput_reader_running is False:
                    break
            
            proc = None
            with self.lock:
                proc = self.procs[0]
            if proc is None: # in case the main process is terminated
                break
            line = proc.stdout.readline()
            if not line:
                break
            line = line.decode()
            if len(line) > 0:
                if "Local-search chosen method is: Solis-Wets (sw)" in line: # miniMDock
                    include_line = True
                    continue
                if include_line is True:
                    self.throughput_queue.put(line)

    def get_throughput_since_last_get(self): 
        lines = list()
        while self.throughput_queue.empty() == False:
            line = self.throughput_queue.get()
            lines.append(line)
        avg_throughput, num_samples = self.calculate_average(lines)
        avg_tp = 1000.0/avg_throughput if avg_throughput != 0.0 else 0.0 # for miniMDock TP is msec/iteration. Convert to iter/sec
        return avg_tp, num_samples
            
    
    def suspend(self):
        if self.debug:
            print(f"Suspending ", flush=True)
        with self.lock:
            if self.procs is not None:
                self.procs[1].suspend()

    def resume(self):
        if self.debug:
            print(f"Resuming", flush=True)
        with self.lock:
            if self.procs is not None:
                self.procs[1].resume()
    
    def terminate(self):
        if self.debug:
            print(f"Terminating ", flush=True)
        with self.lock:
            self.throughput_reader_running = False

        if self.throughput_reader_thread is not None:
            self.throughput_reader_thread.join()

        if self.procs is not None:
            if self.procs[1].is_running():
                self.procs[1].kill()
                # self.proc_log_file.close()
                # self.proc_log_file = None
        with self.lock:
            self.throughput_reader_thread = None
            self.procs = None

        avg_tp, n_samples= self.get_throughput_since_last_get()
        return {"total_avg": avg_tp, "num_iterations": n_samples, "unit": "iteration_per_sec"}


class BatchRemoteRunner:
    be_runners = dict()
    be_runners_stats = dict()
    socket_poller = None
    subscriber_socket = None
    wrote_state_header = False
    def __init__(self, control_ip: str, control_port: str = "45678", app_name: str = "miniMDock"):
        self.ctx = None
        self.app_name = app_name
        self.control_port = control_port
        self.control_ip = control_ip
        self.subscriber_socket, self.socket_poller = self.setup_socket()

    def setup_socket(self):
        self.ctx = zmq.Context.instance()
        publisher = None
        poller = None
        print(f"Connecting ... ", end="")
        try:
            publisher = self.ctx.socket(zmq.SUB)
            publisher.connect(f"tcp://{self.control_ip}:{self.control_port}")
            publisher.setsockopt_string(zmq.SUBSCRIBE, self.app_name)
            # publisher.setsockopt(zmq.CONFLATE, 1)
            print(f"(channel subscribed: {self.app_name}) ", end="")
            poller = zmq.Poller()
            poller.register(publisher, zmq.POLLIN)
            print("Success!")
            
            return publisher, poller
        except Exception as e:
            print(f"Failed! error: {e}")
        return None, None

    def start(self):
        print("Remote BERunner running...", flush=True)
        while(True):
            socks = dict(self.socket_poller.poll(10))
            if self.subscriber_socket in socks and socks[self.subscriber_socket] == zmq.POLLIN:
                msg = None
                try:
                    msg = self.subscriber_socket.recv_string()
                except zmq.ZMQError as e:
                    if e.errno == zmq.ETERM:
                        print("ZMQ socket interrupted/terminated, Quitting...")
                    else:
                        print(f"ZMQ socket error: {e}, Quitting...")
                    break
                # It should not be None here (just in case)
                if msg is None:
                    continue
                print(f"BERunner recvd: {msg}")
                splitted = msg.split(":")
                cmd, args = splitted[0], splitted[1:]
                if cmd == "start": #start:
                    continue
                if cmd == "add_gpu": #add_gpu:gpu
                    gpu = args[0]
                    if gpu not in self.be_runners:
                        self.be_runners[gpu] = BatchRunner()
                        self.be_runners[gpu].run(gpu)
                    else:
                        print(f"be_runner already running on gpu {gpu}, ignoring msg", flush=True)
                elif cmd == "remove_gpu": #remove_gpu:gpu
                    gpu = args[0]
                    if gpu in self.be_runners == False:
                        out = self.be_runners[gpu].terminate()
                        del self.be_runners[gpu]
                        self.be_runners_stats[gpu] = out
                    else:
                        print(f"be_runner does not have be running on gpu {gpu}, ignoring msg", flush=True)
                
                elif cmd == "pause_gpu": #pause_gpu:gpu
                    gpu= args[0]
                    if gpu in self.be_runners:
                        self.be_runners[gpu].suspend()
                    else:
                        print(f"be_runner does not have be running on gpu {gpu}, ignoring msg", flush=True)
                elif cmd == "resume_gpu": #resume_gpu:gpu
                    gpu= args[0]
                    if gpu in self.be_runners:
                        self.be_runners[gpu].resume()
                    else:
                        print(f"be_runner does not have be running on gpu {gpu}, ignoring msg", flush=True)
                elif cmd == "stop": #stop:key1:value1:key2:value2:... (no trailing colon)
                    if len(self.be_runners) > 0:
                        gpus_list = list(self.be_runners.keys())
                        for gpu in gpus_list:
                            out = self.be_runners[gpu].terminate()
                            del self.be_runners[gpu]
                            self.be_runners_stats[gpu] = out
                    if len(args) > 1:
                        # create dict of args
                        args_dict = dict()
                        for i in range(0, len(args)-1, 2):
                            args_dict[args[i]] = args[i+1]
                        stat_file_name = args_dict['stat_file']
                        del args_dict['stat_file']
                        self.dump_stats(args_dict, stat_file_name)
                    else:
                        print(f"be_runner does not have any be running processes, ignoring msg", flush=True)
                    break
                else:
                    print(f"received unsupported command: {cmd}", flush=True)

        # in case runners exist after finish message.
        if len(self.be_runners) > 0:
            for gpu in self.be_runners:
                out = self.be_runners[gpu].terminate()
                del self.be_runners[gpu]
                self.be_runners_stats[gpu] = out
        print("Remote BERunner shut down!", flush=True)

    def dump_stats(self, args, file_name: str):
        print(f"Saving stats to: {file_name}")
        # Create path of stats
        path = file_name[:file_name.rfind("/")]
        os.makedirs(path, exist_ok=True)
        # create stats file if does not exists, if exists, append to it.
        create = True
        if os.path.isfile(file_name):
                create = False
        f = open(file_name, 'w' if create else 'a')
        if create:
            header = "gpu,avg_through_put,iterations,unit"
            for arg_name in list(args.keys()):
                header += ","+str(arg_name)
            header += "\n"
            f.write(header)

        for gpu in self.be_runners_stats:
            gpu_out = self.be_runners_stats[gpu]
            avg_tp = gpu_out["total_avg"]
            iterations = gpu_out["num_iterations"]
            unit = gpu_out["unit"]
            f.write(f"{gpu},{avg_tp},{iterations},{unit}")
            args_str = ""
            for arg_name in list(args.keys()):
                args_str += f",{args[arg_name]}"
            args_str +=  "\n"
            f.write(args_str)
        f.close()
        print(f"Saving stats done")

def test_batch_runner():
    import sys
    if len(sys.argv) < 2:
        print("usage: python batch_runner.py [workloads run command with abs path]")
        return -1
    
    cmd = sys.argv[1]
    print(f"cmd: {cmd}")
    f = BatchRunner(cmd, False)
    import threading
    f.run()
    print("run for 10 sec", flush=True)
    # while True:
    
    time.sleep(10)
    print(f"throughput: {f.get_throughput_since_last_get()}", flush=True)
    print("suspending for 2 sec", flush=True)
    f.suspend()
    time.sleep(2)
    print(f"throughput: {f.get_throughput_since_last_get()}", flush=True)
    print("resuming for 10 sec", flush=True)
    f.resume()
    time.sleep(10)
    print(f"throughput: {f.get_throughput_since_last_get()}", flush=True)
    print("running for 10 sec", flush=True)
    time.sleep(10)

    print(f"termination: {f.terminate()}", flush=True)
    
def test_remote_batch_runner():
    stats_file = "/workspace/workloads/remote_be_runner_stats_test"
    runner = BatchRemoteRunner(
        control_ip="localhost",
        control_port="45678",
        app_name="miniMDock",
    )
    print("Starting...")
    runner.start()
    print("Stopped by remote")

if __name__ == "__main__":
    # test_batch_runner()
    test_remote_batch_runner()
