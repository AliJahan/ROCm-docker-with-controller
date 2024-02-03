import time
from power_collector.power_collector import PowerCollector
from workload_runners.workload_runner_remote import RemoteDockerRunner, RemoteWorkloadRunner

class RemoteExperimentRunner:
    power_collector = None
    resource_manager = None
    docker_runner = None
    workload_runner = None
    resource_controller = None
    dockers_to_cleanup = list()
    def __init__(
            self,
            lc_load_trace: str = "",
            target_ip: str = "172.20.0.9",
            remote_ip: str = "172.20.0.6",
            remote_resource_ctl_port: str = "2000",
            remote_workload_ctl_port: str = "3000",
            target_docker_ctrl_port: str = "4000",
            target_power_broadcaster_port: str = "6000",
            
        ):
        self.target_ip = target_ip
        self.remote_ip = remote_ip
        self.lc_load_trace = lc_load_trace
        self.remote_resource_ctl_port = remote_resource_ctl_port
        self.remote_workload_ctl_port = remote_workload_ctl_port
        self.target_docker_ctrl_port = target_docker_ctrl_port
        self.target_power_broadcaster_port = target_power_broadcaster_port

        self.docker_runner = RemoteDockerRunner(
            remote_ip=remote_ip,
            target_ip=target_ip,
            target_docker_control_port=target_docker_ctrl_port,
            remote_workload_control_port=remote_workload_ctl_port
        )

        self.workload_runner = RemoteWorkloadRunner(
            remote_ip=remote_ip,
            target_ip=target_ip,
            remote_workload_control_port=remote_workload_ctl_port,
            client_trace_file=lc_load_trace
        )
    def cleanup(self):
        print("Cleaning up the experiment")
        if self.power_collector is not None:
            del self.power_collector
        self.docker_runner.cleanup()

    def setup_server(self):
        print(f"Staring power-broadcaster image on target server: {self.target_ip} ", flush=True)
        assert self.docker_runner.start_docker("power-broadcaster") == True, "Failed! stopping experiments!"
    def start_power_collection(self):
        assert self.power_collector is None, "power collector is alreary running"
        self.power_collector = PowerCollector(
            power_broadcaster_ip=self.target_ip,
            power_broadcaster_port=self.target_power_broadcaster_port,
            collection_interval_sec=1
        )
        self.power_collector.start()

    def stop_power_collection(self):
        assert self.power_collector is not None, "power collector is not running"
        power = self.power_collector.get_all_powers()
        del self.power_collector
        self.power_collector = None
        return power

    def start(self):
        args = dict()
        args['model'] = "resnet152"
        args['gpu'] = "1"
        args['batch_size'] = "8"

        self.setup_server()
        self.start_power_collection()
        time.sleep(30)
        print("Idle powers:")
        print(self.stop_power_collection())
        assert self.docker_runner.start_docker("Inference-Server")== True, "Could not start Inference-Servre"
        
        print("Starting inference server..." , flush=True)
        print(self.workload_runner.start(is_be_wl=False))
        print("Adding gpu ..." , flush=True)
        print(self.workload_runner.add_gpu(is_be_wl=False, args=args))
        self.start_power_collection()
        print(
            self.workload_runner.run_lc_client(
                warmp_first=True,
                gpus=1,
                max_rps_per_gpu=150,
                trace_unit_sec=60
            )
        )
        print("Serving powers:")
        print(self.stop_power_collection())
        self.cleanup()

    def __del__(self):
        self.cleanup()

if __name__ == "__main__":
    import time
    remote_runner = RemoteExperimentRunner()
    remote_runner.start()
    print("Done with experiment waiting 10 sec")
    time.sleep(10)

#IP=`ip r | head -n 1 | awk n 1 | awk '{p`intf $3}'