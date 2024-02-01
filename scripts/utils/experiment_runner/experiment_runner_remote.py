import yaml
from power_collector.power_collector import PowerCollector
from workload_runners.workload_runner_remote import RemoteWorkloadsRunner

class RemoteExperimentRunner:
    power_collector = None
    resource_manager = None
    workload_runner = None
    def __init__(
            self,
            lc_load_trace: str = "/tmp/trace",
            target_ip: str = "172.20.0.9",
            remote_ip: str = "172.20.0.6",
            target_resource_ctl_port: str = "2000",
            target_workload_ctrl_port: str = "4000",
            target_power_broadcaster_port: str = "6000",
            
        ):
        self.power_collector = PowerCollector(
            power_broadcaster_ip=target_ip,
            power_broadcaster_port=target_power_broadcaster_port,
            collection_interval_sec=1)
    
        self.workload_runner = RemoteWorkloadsRunner(
            remote_ip=remote_ip,
            target_ip=target_ip,
            workload_ctrl_port=target_workload_ctrl_port,
            client_trace_file=lc_load_trace
        )
    
    def start(self):
        self.workload_runner.run_docker("test_wl")
        

if __name__ == "__main__":
    remote_runner = RemoteExperimentRunner()
    remote_runner.start()
#IP=`ip r | head -n 1 | awk n 1 | awk '{p`intf $3}'