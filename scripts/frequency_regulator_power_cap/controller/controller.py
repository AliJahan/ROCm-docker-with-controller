import sys
sys.path.insert(1, '../../') # for utils TODO: make it a wheel

from utils.stat_collectors.lc_load_collector import LCLoadCollector
from utils.resource_managers.resource_manager import GPUResourceManager
from utils.workload_runners.workload_runner_remote import RemoteDockerRunner, RemoteWorkloadRunner


class FRWorkloadController:
    def __init__(
            self,
            lc_workload_name: str = "Inference-Server",
            be_workload_name: str= "miniMDock",
            target_ip: str = "172.20.0.9",
            remote_ip: str = "172.20.0.6",
            remote_resource_ctl_port: str = "5000",
            remote_workload_ctl_port: str = "3000",
            target_docker_ctrl_port: str = "4000"
        ):

        self.RESULTS_DIR = "/workspace/fr_results"
        self.LOGS_DIR = "/workspace/experiment_logs"
        self.target_ip = target_ip
        self.remote_ip = remote_ip
        self.remote_resource_ctl_port = remote_resource_ctl_port
        self.remote_workload_ctl_port = remote_workload_ctl_port
        self.target_docker_ctrl_port = target_docker_ctrl_port

        self.docker_runner = RemoteDockerRunner(
            remote_ip=remote_ip,
            target_ip=target_ip,
            target_docker_control_port=target_docker_ctrl_port,
            remote_workload_control_port=remote_workload_ctl_port,
            remote_resource_ctl_port=remote_resource_ctl_port
        )

        self.workload_runner = RemoteWorkloadRunner(
            remote_ip=self.remote_ip,
            target_ip=self.target_ip,
            remote_workload_control_port=self.remote_workload_ctl_port,
            lc_workload_name=lc_workload_name,
            be_workload_name=be_workload_name
        )
        
        self.resource_controller = GPUResourceManager(
            remote_control_ip=remote_ip,
            remote_control_port=remote_resource_ctl_port
        )