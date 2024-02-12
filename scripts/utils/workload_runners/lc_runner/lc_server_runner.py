import os
import time 
import subprocess
import psutil
import zmq

from lc_runner.lc_controller import LCController

class LCServerRunner:
    logs_dir = "/workspace/logs"
    proc_log_file = None
    def __init__(self, debug = False):
        self.proc = None
        self.debug = debug
        self.controller = LCController()
        # threading.Thread.__init__(self)
    
    def run_server(self, cmd):
        print("Running the LC server ... ", flush=True, end="")
        # CUMASKING_CONTROLLER_LOG is set for cumasking controller (required for rocr to activate cumasking)
        env = {
            **os.environ,
            "CUMASKING_CONTROLLER_LOG": f"{self.logs_dir}/lc_rocr",
            # "OMP_NUM_THREADS": f'{int(ceil(n_gpus/2))}',
            "OMP_NUM_THREADS": f'{int(1)}',
            # "OMP_THREAD_LIMIT": 
            "OMP_DYNAMIC": "false",
            # ROCM
            # "ROCM_PATH": "/opt/rocm",
            # "ROCM_RPATH":"/opt/rocm/lib",
            # "HIP_PATH":"/opt/rocm/hip",
            # "HIP_PLATFORM": "amd",
            # "HIP_RUNTIME": "rocclr",
            # "HIP_COMPILER": "clang",
            # "ROCM_TOOLKIT_PATH": "/opt/rocm",
            # "DEVICE_LIB_PATH": "/opt/rocm/amdgcn/bitcode/",
            # "PATH": path_all,
            # "LD_LIBRARY_PATH": ld_all,
            # "HIP_HIDDEN_FREE_MEM": "320",
            # "AMDDeviceLibs_DIR": "/opt/rocm/rocdl",
            # "amd_comgr_DIR": "/opt/rocm/comgr",
        }
        self.proc_log_file = open(f"{self.logs_dir}/lc_run_log", 'w')
        p = subprocess.Popen(
            cmd.split(" "),
            env=env,
            stdout=self.proc_log_file,
            stdin=subprocess.PIPE,
            stderr=self.proc_log_file
        )
        self.proc = (p, psutil.Process(pid=p.pid))
        print("Done!", flush=True)
        time.sleep(2)
    
    def set_batch_size(self, model, batch_size):
        print(f"Setting batch size for model {model} to {batch_size}", flush=True)
        return self.controller.set_batch_size(model, batch_size)

    def add_worker(self, model, gpu):
        print(f"Adding worker for model {model} on gpu {gpu}", flush=True)
        return self.controller.add_gpu(model, gpu)

    def remove_worker(self, model, gpu):
        print(f"Removing worker for model {model} from gpu {gpu}", flush=True)
        return self.controller.remove_gpu(model, gpu)

    def pause_worker(self, model, gpu):
        print(f"Pausing worker for model {model} on gpu {gpu}", flush=True)
        return self.controller.pause_gpu(model, gpu)

    def resume_worker(self, model, gpu):
        print(f"Resuming worker for model {model} on gpu {gpu}", flush=True)
        return self.controller.resume_gpu(model, gpu)
    
    def stop(self):
        if self.debug:
            print(f"Terminating server ... ", flush=True, end="")
    
        if self.proc is not None:
          self.proc[1].kill()
          self.proc_log_file.close()
          self.proc_log_file = None
        #   output, _= self.proc[0].communicate()
    
        self.proc = None
        
        if self.debug:
            print(f"Done!", flush=True, end="")

class LCServerRunnerWrapper:
    server_cmd = "/workloads/Inference-server/build/bin/server -m /workloads/model_repo/ -e"
    server_runner = None
    def start_server(self):
        if self.server_runner is not None:
            self.server_runner.stop()
        if self.server_runner is None:
            self.server_runner = LCServerRunner(True)
            self.server_runner.run_server(self.server_cmd)

    def add_gpu(self, model, gpu, batch_size):
        return self.server_runner.add_worker(model, gpu) and self.server_runner.set_batch_size(model, batch_size)

    def remove_gpu(self, model, gpu):
        return self.server_runner.remove_worker(model, gpu)

    def stop_server(self):
        if self.server_runner is not None:
            self.server_runner.stop()
        self.server_runner = None

    def pause_worker(self, model, gpu):
        if self.server_runner is not None:
            return self.server_runner.pause_worker(model, gpu)
        return False
    
    def resume_worker(self, model, gpu):
        if self.server_runner is not None:
            return self.server_runner.resume_worker(model, gpu)
        return False

class LCRemoteRunner:
    lc_runner = None
    socket_poller = None
    subscriber_socket = None
    def __init__(self, control_ip: str, control_port: str, app_name: str = "Inference-Server"):
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
        print("Remote LCRunner running...", flush=True)
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

                print(f"LCRunner recvd: {msg}")
                splitted = msg.split(":")
                cmd, args = splitted[0], splitted[1:]
                if cmd == "start": #start:
                    if self.lc_runner is None:
                        self.lc_runner = LCServerRunnerWrapper()
                        self.lc_runner.start_server()
                    else:
                        print("lc_runner already running, ignoring msg", flush=True)

                elif cmd == "add_gpu": #add_gpu:model:gpu:batch_size
                    model, gpu, batch_size = args
                    if self.lc_runner is not None:
                        self.lc_runner.add_gpu(model=model, gpu=gpu, batch_size=batch_size)
                    else:
                        print("lc_runner is not running, ignoring msg", flush=True)
                elif cmd == "remove_gpu": #remove_gpu:model:gpu
                    model, gpu = args
                    if self.lc_runner is not None:
                        self.lc_runner.remove_gpu(model=model, gpu=gpu)
                    else:
                        print("lc_runner is not running, ignoring msg", flush=True)
                elif cmd == "pause_gpu": #pause_gpu:model:gpu
                    model, gpus= args
                    if self.lc_runner is not None:
                        self.lc_runner.pause_worker(model=model, gpus=gpus)
                    else:
                        print("lc_runner is not running, ignoring msg", flush=True)
                elif cmd == "resume_gpu": #pause_gpu:model:gpu
                    model, gpus= args
                    if self.lc_runner is not None:
                        self.lc_runner.resume_worker(model=model, gpus=gpus)
                    else:
                        print("lc_runner is not running, ignoring msg", flush=True)
            
                elif cmd == "stop":
                    if self.lc_runner is not None:
                        self.lc_runner.stop_server()
                        self.lc_runner = None
                    print(f"received stop command, shutting down", flush=True)
                    break

                else:
                    print(f"received unsupported command: {msg}", flush=True)

        if self.lc_runner is not None:
            self.lc_runner.stop_server()
            self.lc_runner = None
        
        print("Remote LCRunner shut down!", flush=True)


def test_lc_server_runner_wrapper():
    f = LCServerRunnerWrapper()
    print("running server with 2 gpus [0,1]")
    f.start_server(model="resnet152", gpus=2, batch_size=8)
    time.sleep(60)
    print("pausing gpu 1")
    f.pause_worker(model="resnet152", gpu=1)
    time.sleep(30)
    print("resuming gpu 1")
    f.resume_worker(model="resnet152", gpu=1)
    time.sleep(30000)
    f.stop_server()

def test_remote_runner():
    runner = LCRemoteRunner(
        control_ip="localhost",
        control_port="45678",
        app_name="Inference-Server",
    )
    print("Starting...")
    runner.start()
    print("Stopped by remote")
    

if __name__ == "__main__":
    # test_lc_server_runner_wrapper()
    test_remote_runner()