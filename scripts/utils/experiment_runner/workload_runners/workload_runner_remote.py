import zmq
import time
from .lc_runner.lc_client_runner import LCClientRunnerWarpper
from .lc_runner.lc_client_runner import LCClientRunnerWarpper

class RemoteDockerRunner:
    SLEEP_AFTER_SEND_MSG_SEC = 5
    docker_controller_socket = None
    remote_header = "remote_docker_runner"
    dockers_to_cleanup = list()
    def __init__(
            self,
            remote_ip: str,
            target_ip: str,
            target_docker_control_port: str,
            remote_workload_control_port: str
    ):
        self.remote_ip = remote_ip
        self.target_ip = target_ip
        self.target_docker_control_port = target_docker_control_port
        self.remote_workload_control_port = remote_workload_control_port
        self.docker_controller_socket = self.setup_socket()

    def setup_socket(self):
        self.ctx = zmq.Context.instance()
        publisher = None
        # poller = None
        print(f"Connecting to {self.target_ip}:{self.target_docker_control_port}... ", end="")
        try:
            publisher = self.ctx.socket(zmq.DEALER)
            publisher.setsockopt_string(zmq.IDENTITY, self.remote_header)
            publisher.connect(f"tcp://{self.target_ip}:{self.target_docker_control_port}")
            # publisher.setsockopt(zmq.CONFLATE, 1)
            # print(f"(channel subscribed: {self.app_name}) ", end="")
            # poller = zmq.Poller()
            # poller.register(publisher, zmq.POLLIN)
            print("Success!")
            return publisher
        except Exception as e:
            print(f"Failed! error: {e}")
        return None

    def send_msg(self, msg):
        print(f"Sending message: {msg} ... ", end="", flush=True)
        rep = False
        try:
            self.docker_controller_socket.send_string(f"{msg}")
            rep = self.docker_controller_socket.recv_string() == "SUCESS"
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                print("FAILED! error: ZMQ socket interrupted/terminated", flush=True)
            else:
                print(f"FAILED! error: ZMQ socket error: {e}", flush=True)

        if rep == True:
            time.sleep(self.SLEEP_AFTER_SEND_MSG_SEC)
            print(f"Done!") 

        return rep

    def start_docker(self, workload):
        msg = f"run:{workload}:{self.remote_ip}:{self.remote_workload_control_port}"
        ret = self.send_msg(msg)
        if ret == True:
            if workload not in self.dockers_to_cleanup:
                self.dockers_to_cleanup.append(workload)
        return ret
        
    def stop_docker(self, workload):
        msg = f"stop:{workload}"
        ret = self.send_msg(msg)
        if ret == True:
            if workload in self.dockers_to_cleanup:
                self.dockers_to_cleanup.remove(workload)
        return ret
    def cleanup(self):
        if len(self.dockers_to_cleanup) == 0:
            return
        for docker in self.dockers_to_cleanup:
            self.stop_docker(docker)
    def __del__(self):
        self.cleanup()

class RemoteWorkloadRunner:
    SLEEP_AFTER_SEND_MSG_SEC = 5
    publisher_socket = None
    be_runner_channel = "miniMDock"
    lc_runner_channel = "Inference-Server"
    def __init__(
            self,
            remote_ip: str,
            target_ip: str,
            remote_workload_control_port: str,
            client_trace_file: str
    ):
        self.remote_ip = remote_ip
        self.target_ip = target_ip
        self.remote_workload_control_port = remote_workload_control_port
        self.client_trace_file = client_trace_file
        self.publisher_socket = self.setup_socket()

    def setup_socket(self):
        self.ctx = zmq.Context.instance()
        publisher = None
        # poller = None
        print(f"Binding to {self.remote_ip}:{self.remote_workload_control_port}... ", end="")
        try:
            publisher = self.ctx.socket(zmq.PUB)
            publisher.bind(f"tcp://*:{self.remote_workload_control_port}")
            print("Success!")
            return publisher
        except Exception as e:
            print(f"Failed! error: {e}")
        return None

    def run_lc_client(self, warmp_first, gpus, max_rps_per_gpu, trace_unit_sec):
        client = LCClientRunnerWarpper(
            num_warmpup_load_steps=4,
            warmup_step_duration_sec=60,
            trace_file=self.client_trace_file,
            gpus=gpus,
            max_rps_per_gpu=max_rps_per_gpu,
            trace_unit_sec=trace_unit_sec
        )

        if warmp_first == True:
            client.warmup(server_ip=self.target_ip)
        
        return client.run(server_ip=self.target_ip)

    def send_msg(self, channel, msg):
        print(f"Sending message: ({channel}) {msg} ... ", end="", flush=True)
        rep = True
        try:
            self.publisher_socket.send_string(f"{channel}", flags=zmq.SNDMORE)
            self.publisher_socket.send_string(f"{msg}")
            # rep = self.publisher_socket.recv_string() == "SUCESS"
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                print("FAILED! error: ZMQ socket interrupted/terminated", flush=True)
            else:
                print(f"FAILED! error: ZMQ socket error: {e}", flush=True)
        
        if rep == True:
            time.sleep(self.SLEEP_AFTER_SEND_MSG_SEC)
            print(f"Done!") 

        return rep
    def get_channel(self, is_be_wl):
        return self.be_runner_channel if is_be_wl == True else self.lc_runner_channel

    def start(self, is_be_wl):
        channel = self.get_channel(is_be_wl=is_be_wl)
        msg = f"start:"
        return self.send_msg(channel, msg)

    def add_gpu(self, is_be_wl, args):
        msg = f"add_gpu:"
        if is_be_wl:
            msg += args["gpu"]
        else:
            msg += args['model']+":"+args['gpu']+":"+args['batch_size']

        channel = self.get_channel(is_be_wl=is_be_wl)
        return self.send_msg(channel, msg)
    
    def pause_gpu(self, is_be_wl, args):
        msg = f"pause_gpu:"
        if is_be_wl:
            msg += args["gpu"]
        else:
            msg += args['model']+":"+args['gpu']

        channel = self.get_channel(is_be_wl=is_be_wl)
        return self.send_msg(channel, msg)
    
    def remove_gpu(self, is_be_wl, args):
        msg = f"remove_gpu:"
        if is_be_wl:
            msg += args["gpu"]
        else:
            msg += args['model']+":"+args['gpu']
        
        channel = self.get_channel(is_be_wl=is_be_wl)
        return self.send_msg(channel, msg)
    
    def finsh_wl(self, is_be_wl, args):
        msg = f"stop:"
        if is_be_wl:
            msg += args['stat_file']

        channel = self.get_channel(is_be_wl=is_be_wl)
        return self.send_msg(channel, msg)
    
    def resume_gpu(self, is_be_wl, args):
        msg = f"resume_gpu:"
        if is_be_wl:
            msg += args['gpu']
        else:
            msg += args['model']+":"+args['gpu']

        channel = self.get_channel(is_be_wl=is_be_wl)
        return self.send_msg(channel, msg)
