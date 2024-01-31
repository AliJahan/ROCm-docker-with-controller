import zmq
import subprocess
import os 

class TargetExperimentRunner:
    subscriber_socket = None
    channel_name = "experiment_runner_target"
    def __init__(
            self,
            project_root_path: str,
            remote_ip: str,
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
        print(f"Binding {self.remote_ip}:{self.remote_port}... ", end="")
        try:
            publisher = self.ctx.socket(zmq.SUB)
            publisher.connect(f"tcp://{self.remote_ip}:{self.remote_port}")
            publisher.subscribe("")
            # publisher.setsockopt(zmq.CONFLATE, 1)
            
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
        print(f"Running target experiment runner channel:{self.channel_name} remote_ip:{self.remote_ip} remote_port:{self.remote_port}")

        while True:
            msg = None
            try:
                msg = self.subscriber_socket.recv_string()
                print("rcvs", flush=True)
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    print("ZMQ socket interrupted/terminated, Quitting...")
                else:
                    print(f"ZMQ socket error: {e}, Quitting...")
                break
            if msg is None:
                continue
            print(f"rcvd mesg: {msg}", flush=True)
            cmd, args = msg.split(":")
            # if cmd == "run":
            #     workload, remote_ip, remote_port = args


if __name__ == "__main__":
    remote_ip = "172.20.0.1"
    remote_port = "4000"
    project_path = "/home/ajaha004/repos/rocr/standalone-docker/ROCm-docker-with-controller/"
    expr_runner = TargetExperimentRunner(project_root_path=project_path, remote_ip=remote_ip, remote_port=remote_port)
    expr_runner.start()