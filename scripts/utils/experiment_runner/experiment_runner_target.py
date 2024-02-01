import zmq
import subprocess
import os 

class TargetExperimentRunner:
    subscriber_socket = None
    supported_images = ["power-broadcaster", "Inference-Server", "miniMDock"]
    ctx = None
    def __init__(
            self,
            project_root_path: str,
            control_port: str
        ):
        self.project_root_path = project_root_path
        self.control_port = control_port
        self.subscriber_socket = self.setup_socket()

    def setup_socket(self):
        self.ctx = zmq.Context.instance()
        publisher = None
        # poller = None
        print(f"Binding docker_runner to localhost:{self.control_port} ... ", end="")
        try:
            publisher = self.ctx.socket(zmq.ROUTER)
            publisher.bind(f"tcp://*:{self.control_port}")
            
            # poller = zmq.Poller()
            # poller.register(publisher, zmq.POLLIN)
            print(f"Success!")
        except Exception as e:
            print(f"Failed! error: {e}")
        return publisher
        
    def reply_res(self, client, res):
        msg = "SUCESS" if res == True else "FAILED!"
        self.subscriber_socket.send(client, flags=zmq.SNDMORE)
        self.subscriber_socket.send_string(msg)

    def verify_image_name(self, image_name):
        if image_name not in self.supported_images:
            print(f"Error: unsupported docker name: {image_name}", end="", flush=True)
            return False
        return True

    def run_docker(self, image_name, remote_ip, remote_port):
        if self.verify_image_name(image_name=image_name) is False:
            return False

        env = {
            **os.environ,
            "CONTROLLER_IP": str(remote_ip),
            "CONTROLLER_PORT": str(remote_port),
            "PROJECT_ROOT": str(self.project_root_path),
            "LOGS_DIR": str(self.project_root_path+"/"+"docker_logs")
        }
        
        cmd = f"{self.project_root_path}/scripts/docker/run_image.sh {image_name}"
        p = subprocess.Popen(
            cmd.split(" "),
            env=env,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        p.wait()
        return True

    def stop_docker(self, image_name):
        if self.verify_image_name(image_name=image_name) is False:
            return False
        
        cmd = f"{self.project_root_path}/scripts/docker/stop_image.sh {image_name}"
        env = {
            **os.environ,
            "PROJECT_ROOT": str(self.project_root_path),
            "LOGS_DIR": str(self.project_root_path+"/"+"docker_logs")
        }
        p = subprocess.Popen(
            cmd.split(" "),
            env=env,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        p.wait()
        return True
    
    def start(self):
        if self.subscriber_socket is None:
            return
        print(f"Running target experiment runner localhost:{self.control_port}")
        while True:
            msg = None
            sender = None
            try:
                sender = self.subscriber_socket.recv()
                msg = self.subscriber_socket.recv_string()
                print(f"rcved from ({sender}): {msg}", flush=True)
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    print("ZMQ socket interrupted/terminated, Quitting...", flush=True)
                else:
                    print(f"ZMQ socket error: {e}, Quitting...", flush=True)
                break
            if msg is None:
                continue
            splitted = msg.split(":")
            cmd, args = splitted[0], splitted[1:]
            if cmd == "run":
                image_name, remote_ip, remote_port = args
                res = self.run_docker(image_name=image_name, remote_ip=remote_ip, remote_port=remote_port)
                self.reply_res(sender, res)
            elif cmd == "stop":
                image_name = args[0]
                res = self.stop_docker(image_name=image_name)
                self.reply_res(sender, res)
            
        print("done!!")

def main():
    control_port = "4000"
    project_path = "/home/ajaha004/repos/rocr/standalone-docker/ROCm-docker-with-controller"
    expr_runner = TargetExperimentRunner(project_root_path=project_path, control_port=control_port)
    expr_runner.start()
    print("@@done")
if __name__ == "__main__":
    main()