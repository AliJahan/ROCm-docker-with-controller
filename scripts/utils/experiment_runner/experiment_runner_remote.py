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

    def start(self):
        self.setup_server()
        
    def __del__(self):
        self.cleanup()

if __name__ == "__main__":
    import time
    remote_runner = RemoteExperimentRunner()
    remote_runner.start()

#IP=`ip r | head -n 1 | awk n 1 | awk '{p`intf $3}'