import time
from power_collector.power_collector import PowerCollector
from workload_runners.workload_runner_remote import RemoteDockerRunner, RemoteWorkloadRunner

class RemoteExperimentRunner:
    power_collector = None
    resource_manager = None
    docker_runner = None
    workload_runner = None
    resource_controller = None
    def __init__(
            self,
            lc_load_trace: str = "/tmp/trace",
            target_ip: str = "172.20.0.9",
            remote_ip: str = "172.20.0.6",
            remote_resource_ctl_port: str = "2000",
            remote_workload_ctl_port: str = "3000",
            target_docker_ctrl_port: str = "4000",
            target_power_broadcaster_port: str = "6000",
            
        ):
        self.target_ip = target_ip
        self.power_collector = PowerCollector(
            power_broadcaster_ip=target_ip,
            power_broadcaster_port=target_power_broadcaster_port,
            collection_interval_sec=1)

        self.docker_runner = RemoteDockerRunner(
            remote_ip=remote_ip,
            target_ip=target_ip,
            target_docker_control_port=target_docker_ctrl_port,
            remote_workload_control_port=remote_workload_ctl_port
        )

        self.workload_runner = RemoteWorkloadRunner(
            remote_ip=remote_ip,
            remote_workload_control_port=remote_workload_ctl_port,
            client_trace_file=lc_load_trace
        )
    def cleanup(self):
        print(f"Stopping power-broadcaster image on target server (just in case)", flush=True)
        assert self.docker_runner.stop_docker("power-broadcaster") == True, "Could not stop"

    def setup_server(self):
        print(f"Staring power-broadcaster image on target server: {self.target_ip} ", flush=True)
        assert self.docker_runner.start_docker("power-broadcaster") == True, "Failed! stopping experiments!"
    def wait_sec(self, wait_time_sec: int):
        time.sleep(wait_time_sec)
    def start(self):
        self.setup_server()
        assert self.docker_runner.start_docker("Inference-Server")== True, "Could not start Inference-Servre"
        args = dict()
        args['model'] = "resnet152"
        args['gpu'] = "1"
        args['batch_size'] = "8"
        print("Starting inference server..." ,end="", flush=True)
        print(self.workload_runner.start(is_be_wl=False))
        self.wait_sec(5)
        print("Adding gpu ..." ,end="", flush=True)
        print(self.workload_runner.add_gpu(is_be_wl=False, args=args))
        self.wait_sec(5)
        # print("Running client ..." ,end="", flush=True)
        # print(self.workload_runner.run_lc_client())
        
    def __del__(self):
        self.cleanup()

if __name__ == "__main__":
    import time
    remote_runner = RemoteExperimentRunner()
    remote_runner.start()
    print("Done with experiment waiting 10 sec")
    time.sleep(10)

#IP=`ip r | head -n 1 | awk n 1 | awk '{p`intf $3}'