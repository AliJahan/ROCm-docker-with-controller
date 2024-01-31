import zmq
from .lc_runner.lc_client_runner import LCClientRunnerWarpper
from .lc_runner.lc_client_runner import LCClientRunnerWarpper

class RemoteWorkloadsRunner:
    publisher_socket = None
    target_runner_channel = "experiment_runner_target"
    be_runner_channel = "miniMDock"
    lc_runner_channel = "Inference-Server"
    def __init__(
            self,
            remote_ip: str,
            target_ip: str,
            workload_ctrl_port: str,
            client_trace_file: str
    ):
        self.remote_ip = remote_ip
        self.target_ip = target_ip
        self.workload_ctrl_port = workload_ctrl_port
        self.client_trace_file = client_trace_file
        self.publisher_socket = self.setup_socket()

    def setup_socket(self):
        self.ctx = zmq.Context.instance()
        publisher = None
        # poller = None
        print(f"Binding {self.remote_ip}:{self.workload_ctrl_port}... ", end="")
        try:
            publisher = self.ctx.socket(zmq.PUB)
            publisher.bind(f"tcp://0.0.0.0:{self.workload_ctrl_port}")
            # publisher.setsockopt(zmq.SUBSCRIBE, bytes(self.app_name.encode('utf-8')))
            # publisher.setsockopt(zmq.CONFLATE, 1)
            # print(f"(channel subscribed: {self.app_name}) ", end="")
            # poller = zmq.Poller()
            # poller.register(publisher, zmq.POLLIN)
            print("Success!")
            return publisher
        except Exception as e:
            print(f"Failed! error: {e}")
        return None

    def run_lc_client(self, warmp_first, gpus, max_rps_per_gpu, trace_unit_sec):
        client = LCClientRunnerWarpper(
            num_warmpup_load_steps=3,
            warmup_step_duration_sec=10,
            trace_file=self.client_trace_file,
            gpus=gpus,
            max_rps_per_gpu=max_rps_per_gpu,
            trace_unit_sec=trace_unit_sec
        )

        if warmp_first == True:
            client.warmup(server_ip=self.target_ip)
        
        return client.run(server_ip=self.target_ip)

    def send_msg(self, channel, msg):
        print(f"Sending message: ({channel}): {msg} ... ", end="")
        try:
            self.publisher_socket.send_string(channel, flags=zmq.SNDMORE)
            self.publisher_socket.send_string(msg)
            print(f"Done!")
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                print("ZMQ socket interrupted/terminated")
            else:
                print(f"ZMQ socket error: {e}")
            return False
        return True
    
    def start_wl(self, is_be_wl, args):
        channel = self.be_runner_channel if is_be_wl == True else self.lc_runner_channel
        msg = f"start:{args}"
        return self.send_msg(channel, msg)
    
    def pause_wl(self, is_be_wl, args):
        channel = self.be_runner_channel if is_be_wl == True else self.lc_runner_channel
        msg = f"pause:{args}"
        return self.send_msg(channel, msg)
    
    def stop_wl(self, is_be_wl, args):
        channel = self.be_runner_channel if is_be_wl == True else self.lc_runner_channel
        msg = f"stop:{args}"
        return self.send_msg(channel, msg)
    
    def finsh_wl(self, is_be_wl, args):
        channel = self.be_runner_channel if is_be_wl == True else self.lc_runner_channel
        msg = f"finish:{args}"
        return self.send_msg(channel, msg)
    
    def resume_wl(self, is_be_wl, args):
        channel = self.be_runner_channel if is_be_wl == True else self.lc_runner_channel
        msg = f"resume:{args}"
        return self.send_msg(channel, msg)
    
    def run_docker(self, worklad):
        msg = f"run:{worklad}:{self.remote_ip}:{self.workload_ctrl_port}"
        return self.send_msg(self.target_runner_channel, msg)
        
    def stop_docker(self, worklad):
        msg = f"stop:{worklad}"
        return self.send_msg(self.target_runner_channel, msg)