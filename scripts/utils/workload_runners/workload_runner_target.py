from .batch_runner.batch_runner import BatchRemoteRunner
from .lc_runner.lc_server_runner import LCRemoteRunner

class WorkloadRunner:
    runner = None
    workload = None
    def __init__(
            self,
            control_ip: str,
            control_port: str,
            workload: str) -> None:
        self.workload = workload
        if workload == "miniMDock":
            self.runner = BatchRemoteRunner(app_name=workload, control_ip=control_ip, control_port=control_port)
        elif workload == "Inference-Server":
            self.runner = LCRemoteRunner(app_name=workload, control_ip=control_ip, control_port=control_port)
        else:
            print(f"Workload not supoprted: {workload}")
    def start(self):
        if self.runner is not None:
            self.runner.start()
        

if __name__ == "__main__":
    import os
    workload = os.getenv("WORKLOAD")
    controller_ip = os.getenv("REMOTE_IP")
    controller_port = os.getenv("WORKLOAD_CONTROLLER_PORT")
    runner = WorkloadRunner(
            control_ip=controller_ip,
            control_port=controller_port,
            workload=workload
        )
    print(f"Starting workload runner with control_ip={controller_ip}, control_port={controller_port}, workload={workload}")
    runner.start()
    print(f"Workload runner shut down!")