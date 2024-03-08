import zmq
import time

from .lc_runner.lc_client_runner import LCClientRunnerWarpper
from .lc_runner.lc_client_runner import LCClientRunnerWarpper

class RemoteDockerRunner:
    SLEEP_AFTER_SEND_MSG_SEC = 2
    docker_controller_socket = None
    remote_header = "remote_docker_runner"
    dockers_to_cleanup = list()
    def __init__(
            self,
            remote_ip: str,
            target_ip: str,
            target_docker_control_port: str,
            remote_workload_control_port: str,
            remote_resource_ctl_port: str,
            print_debug_info = True,
            simulate = False
    ):
        
        self.simulate = simulate
        self.print_debug_info = print_debug_info
        self.remote_ip = remote_ip
        self.target_ip = target_ip
        self.target_docker_control_port = target_docker_control_port
        self.remote_workload_control_port = remote_workload_control_port
        self.remote_resource_ctl_port = remote_resource_ctl_port
        self.docker_controller_socket = self.setup_socket()

    def setup_socket(self):
        if self.simulate:
            return None
        self.ctx = zmq.Context.instance()
        publisher = None
        # poller = None
        if self.print_debug_info:
            print(f"\t-[RemoteDockerRunner.setup_socket]: Connecting to {self.target_ip}:{self.target_docker_control_port}... ", end="")
        try:
            publisher = self.ctx.socket(zmq.DEALER)
            publisher.setsockopt_string(zmq.IDENTITY, self.remote_header)
            publisher.connect(f"tcp://{self.target_ip}:{self.target_docker_control_port}")
            if self.print_debug_info:
                print("Success!")
            return publisher
        except Exception as e:
            print(f"Failed! error: {e}")
        return None

    def send_msg(self, msg):
        if self.print_debug_info:
            print(f"\t-[RemoteDockerRunner.send_msg]: Sending message: {msg} ... ", end="", flush=True)
            print(f"Done!", flush=True)

        if self.simulate:
            return True
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
            if self.print_debug_info:
                print(f"Done!") 

        return rep

    def start_docker(self, workload):
        msg = f"run:{workload}:{self.remote_ip}:{self.remote_workload_control_port}:{self.remote_resource_ctl_port}"
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
    def __init__(
            self,
            remote_ip: str,
            target_ip: str,
            remote_workload_control_port: str,
            lc_workload_name: str = "Inference-Server",
            be_workload_name: str = "miniMDock",
            wait_after_send = True,
            print_debug_info = True,
            simulate = False
    ):
        self.simulate = simulate
        self.print_debug_info = print_debug_info
        self.remote_ip = remote_ip
        self.target_ip = target_ip
        self.be_workload_name = be_workload_name
        self.lc_workload_name = lc_workload_name
        self.remote_workload_control_port = remote_workload_control_port
        self.wait_after_send = wait_after_send
        self.publisher_socket = self.setup_socket()

    def setup_socket(self):
        if self.simulate:
            return None
        self.ctx = zmq.Context.instance()
        publisher = None
        # poller = None
        if self.print_debug_info:
            print(f"\t-[RemoteWorkloadRunner]: Binding to {self.remote_ip}:{self.remote_workload_control_port}... ", end="")
        try:
            publisher = self.ctx.socket(zmq.PUB)
            publisher.bind(f"tcp://*:{self.remote_workload_control_port}")
            if self.print_debug_info:
                print("Success!")
            return publisher
        except Exception as e:
            print(f"Failed! error: {e}")
        return None

    def run_lc_client(self, warmp_first, num_warmpup_load_steps, warmup_step_duration_sec, gpus, max_rps_per_gpu, trace_file, trace_unit_sec, no_run = False):
        if self.simulate:
            return None
        client = LCClientRunnerWarpper(
            num_warmpup_load_steps=num_warmpup_load_steps,
            warmup_step_duration_sec=warmup_step_duration_sec,
            trace_file=trace_file,
            gpus=gpus,
            max_rps_per_gpu=max_rps_per_gpu,
            trace_unit_sec=trace_unit_sec
        )

        if warmp_first == True:
            client.warmup(server_ip=self.target_ip)

        if no_run is True:
            return None
        return client.run(server_ip=self.target_ip)

    def send_msg(self, channel, msg):
        if self.print_debug_info:
            print(f"\t-[RemoteWorkloadRunner]: sending message: ({channel}) {msg} ... ", end="", flush=True)
        if self.simulate:
            print(f"Done!", flush=True) 
            return True
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
            rep = False

        if rep == True:
            if self.wait_after_send:
                time.sleep(self.SLEEP_AFTER_SEND_MSG_SEC)
            if self.print_debug_info:
                print(f"Done!", flush=True) 
        return rep
    def get_channel(self, is_be_wl):
        return self.be_workload_name if is_be_wl == True else self.lc_workload_name

    def start(self, is_be_wl):
        channel = self.get_channel(is_be_wl=is_be_wl)
        msg = f"start:"
        return self.send_msg(channel, msg)

    def add_gpu(self, is_be_wl, args):
        msg = f"add_gpu:"
        if is_be_wl:
            msg += str(args["gpu"])
        else:
            msg += str(args['model'])+":"+str(args['gpu'])+":"+str(args['batch_size'])

        channel = self.get_channel(is_be_wl=is_be_wl)
        res = self.send_msg(channel, msg)
        # time.sleep(10) # sleep more for adding gpu since the worker performs a dry run upon instanciation
        return res
    
    def pause_gpu(self, is_be_wl, args):
        msg = f"pause_gpu:"
        if is_be_wl:
            msg += str(args["gpu"])
        else:
            msg += str(args['model'])+":"+str(args['gpu'])

        channel = self.get_channel(is_be_wl=is_be_wl)
        return self.send_msg(channel, msg)
    
    def remove_gpu(self, is_be_wl, args):
        msg = f"remove_gpu:"
        if is_be_wl:
            msg += str(args["gpu"])
        else:
            msg += str(args['model'])+":"+str(args['gpu'])
        
        channel = self.get_channel(is_be_wl=is_be_wl)
        return self.send_msg(channel, msg)
    
    def finsh_wl(self, is_be_wl, args):
        msg = f"stop:"
        if is_be_wl:
            args_list = list(args.keys())
            # append key:value
            for arg_key in args_list:
                msg += arg_key+":"+str(args[arg_key])+":"
            msg = msg[:-1]

        channel = self.get_channel(is_be_wl=is_be_wl)
        return self.send_msg(channel, msg)
    
    def resume_gpu(self, is_be_wl, args):
        msg = f"resume_gpu:"
        if is_be_wl:
            msg += str(args['gpu'])
        else:
            msg += str(args['model'])+":"+str(args['gpu'])

        channel = self.get_channel(is_be_wl=is_be_wl)
        return self.send_msg(channel, msg)
