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
            target_ip: str,
            target_port: str
        ):
        self.project_root_path = project_root_path
        self.target_ip = target_ip
        self.target_port = target_port
        self.subscriber_socket = self.setup_socket()

    def setup_socket(self):
        self.ctx = zmq.Context.instance()
        print("Success!", flush=True)
        publisher = None
        # poller = None
        print(f"Binding *:{self.target_port}... ", end="")
        try:
            publisher = self.ctx.socket(zmq.SUB)
            publisher.bind(f"tcp://*:{self.target_port}")
            publisher.setsockopt(zmq.SUBSCRIBE, b"")
            publisher.setsockopt(zmq.CONFLATE, 0)
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
        print(f"Running target experiment runner channel:{self.channel_name} ip:{self.target_ip} port:{self.target_port}")
        print(self.subscriber_socket)
        print(type(self.subscriber_socket))
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
    print("done")

if __name__ == "__main__":
    remote_ip = "172.20.0.9"
    remote_port = "4000"
    project_path = "/home/ajaha004/repos/rocr/standalone-docker/ROCm-docker-with-controller/"
    expr_runner = TargetExperimentRunner(project_root_path=project_path, target_ip=remote_ip, target_port=remote_port)
    expr_runner.start()