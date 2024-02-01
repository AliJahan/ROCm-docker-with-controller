import zmq
import subprocess
import os 

class TargetExperimentRunner:
    subscriber_socket = None
    channel_name = "target"
    ctx = None
    def __init__(
            self,
            project_root_path: str,
            rempte_ip: str,
            remote_port: str
        ):
        self.project_root_path = project_root_path
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.subscriber_socket = self.setup_socket()

    def setup_socket(self):
        self.ctx = zmq.Context.instance()
        print("Success!", flush=True)
        publisher = None
        # poller = None
        print(f"Connecting {self.remote_ip}:{self.remote_port}... ", end="")
        try:
            publisher = self.ctx.socket(zmq.SUB)
            publisher.connect(f"tcp://{self.remote_ip}:{self.remote_port}")
            publisher.setsockopt(zmq.SUBSCRIBE, b"")
            # publisher.setsockopt(zmq.CONFLATE, 0)
            # publisher.setsockopt(zmq.LINGER, 0)
            
            # poller = zmq.Poller()
            # poller.register(publisher, zmq.POLLIN)
            print(f"Success (channel: {self.channel_name})!")
            return publisher
        except Exception as e:
            print(f"Failed! error: {e}")
        return None
    def run_docker(self, workload, remote_ip, remote_port):
        env = {
            **os.environ,
            "WORKLOAD": str(workload),
            "CONTROLLER_IP": str(remote_ip),
            "CONTROLLER_PORT": str(remote_port)
        }
        cmd = f"{self.project_root_path}/scripts/docker/run_image.sh "
        p = subprocess.Popen(
            cmd.split(" "),
            env=env,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    def start(self):
        
        if self.subscriber_socket is None:
            return
        print(f"Running target experiment runner channel:{self.channel_name} ip:{self.remote_ip} port:{self.remote_port}")
        while True:
            msg = None
            try:
                print("rcving", flush=True)
                msg = self.subscriber_socket.recv_string()
                print("rcvs", flush=True)
                # msg = self.subscriber_socket.recv_string()
                print("rcvs", flush=True)
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    print("ZMQ socket interrupted/terminated, Quitting...", flush=True)
                else:
                    print(f"ZMQ socket error: {e}, Quitting...", flush=True)
                break
            if msg is None:
                continue
            print(f"rcvd mesg: {msg}", flush=True)
            cmd, args = msg.split(":")
            # if cmd == "run":
            #     workload, remote_ip, remote_port = args
    print("done")

if __name__ == "__main__":
    remote_ip = "172.20.0.6"
    remote_port = "4000"
    project_path = "/home/ajaha004/repos/rocr/standalone-docker/ROCm-docker-with-controller/"
    expr_runner = TargetExperimentRunner(project_root_path=project_path, target_ip=remote_ip, target_port=remote_port)
    expr_runner.start()
    print("done")