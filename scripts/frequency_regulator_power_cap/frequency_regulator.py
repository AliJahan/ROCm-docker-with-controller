import os
import sys
sys.path.insert(1, '../') # for utils TODO: make it a wheel
# workload control
from utils.stat_collectors.power_collector import PowerCollector


from regulation_signal_tools.regulation_signal_sampler import RSSampler
from scripts.frequency_regulator_power_cap.power_model_tools.fr_optimizer import Optimize
from controller.controller import 
from power_model_tools.power_model import PowerModel

class FrequencyRegulator:
    power_collector = None
    new_avg_power = None
    orig_avg_power = None
    def __init__(
            self,
            regulation_signal_file: str,
            lc_workload_name: str,
            lc_qos_msec: int,
            lc_qos_metric: str,
            num_system_gpus: int,
            lc_max_rps_per_gpu: int,
            lc_power_profile_dir_path: str,
            lc_load_pct_trace_file: str,
            be_workload_name: str,
            be_power_profile_dir_path: str,
            target_ip: str = "172.20.0.9",
            remote_ip: str = "172.20.0.6",
            remote_resource_ctl_port: str = "5000",
            remote_workload_ctl_port: str = "3000",
            target_docker_ctrl_port: str = "4000",
            target_power_broadcaster_port: str = "6000"
        ):
        # LC information
        self.lc_workload_name = lc_workload_name
        self.lc_qos_msec = lc_qos_msec
        self.lc_qos_metric = lc_qos_metric
        self.num_system_gpus = num_system_gpus
        self.lc_max_rps_per_gpu = lc_max_rps_per_gpu
        self.lc_power_profile_dir_path = lc_power_profile_dir_path
        self.lc_load_pct_trace_file = lc_load_pct_trace_file
        
        # BE information
        self.be_workload_name = be_workload_name
        self.be_power_profile_dir_path = be_power_profile_dir_path

        # Controlling information
        self.target_ip = target_ip
        self.remote_ip = remote_ip
        self.target_power_broadcaster_port = target_power_broadcaster_port
        
        self.power_model = PowerModel(
            lc_qos_msec=lc_qos_msec,
            lc_qos_metric=lc_qos_metric,
            num_system_gpus=num_system_gpus,
            lc_max_rps_per_gpu=lc_max_rps_per_gpu,
            lc_power_profile_dir_path=lc_power_profile_dir_path,
            lc_load_pct_trace_file=lc_load_pct_trace_file,
            be_power_profile_dir_path=be_power_profile_dir_path
        )
        
        # self.workload_controller = FRWorkloadController(
        #     lc_workload_name=lc_workload_name,
        #     be_workload_name=be_workload_name,
        #     remote_ip=remote_ip,
        #     target_ip=target_ip,
        #     target_docker_ctrl_port=target_docker_ctrl_port,
        #     remote_resource_ctl_port=remote_resource_ctl_port,
        #     remote_workload_ctl_port=remote_workload_ctl_port
        # )

        self.rs_sampler = RSSampler(
            rs_file_path=regulation_signal_file
        )

    def start(
            self, 
            electricity_cost: float,
            regulation_up_reward: float,
            regulation_down_reward: float
        ):
        optimization_res = self.power_model.optimized_for_fr(
            elec_cost=electricity_cost,
            reg_down=regulation_down_reward,
            reg_up=regulation_up_reward
        )
        # if not participating...
        if optimization_res['powers']['offset_power'] is 0:
            optimization_res['metrics'] = {
                'perf_score': None,
                'lc_qos_pass': True,
                'be_tp': 1.0
            }
            return optimization_res

        avg_power = optimization_res['powers']['offset_power']
        max_decrease_watt = optimization_res['powers']["regulation"]['reg_up']
        max_increase_watt = optimization_res['powers']["regulation"]['reg_down']
        self.setup_server()
        self.warmp()
        self.start_client()
        self.start_power_collection()
        for rs_val in self.rs_sampler.sample(450)[0]:
            next_power = max_increase_watt if rs_val > 0 else -max_decrease_watt
            next_power *= rs_val
            next_power += avg_power
            curr_power = self.get_current_remote_power()
            

    def start_power_collection(self):
        assert self.power_collector is None, "[FrequencyRegulator/start_power_collection]: power collector is alreary running"
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

    def get_current_remote_power(self):
        assert self.power_collector is not None, "power collector is not running"
        return self.power_collector.get_cur_power()


def test_frequency_regulator():
    fr = FrequencyRegulator(
            regulation_signal_file="",
            electricity_cost=1,
            regulation_reward=1,
            lc_workload_name="Inference-Server",
            lc_qos_msec=40,
            lc_qos_metric="p95th",
            lc_max_rps_per_gpu=95,
            lc_power_profile_dir_path="/home/ajaha004/repos/ROCm-docker-with-controller/results",
            lc_load_pct_trace_file="",
            be_workload_name="miniMDock",
            be_power_profile_dir_path="/home/ajaha004/repos/ROCm-docker-with-controller/results"
    )


if __name__ == "__main__":
    test_frequency_regulator()