import os
import sys
import time
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # for utils TODO: make it a wheel
# workload control
from utils.stat_collectors.power_collector import PowerCollector


from regulation_signal_tools.regulation_signal_sampler import RSSampler
from controller.controller import FRSingleGPUWorkloadController
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
            lc_load_pct_time_unit_sec: str,
            be_workload_name: str,
            be_power_profile_dir_path: str,
            target_ip: str = "172.20.0.9",
            remote_ip: str = "172.20.0.6",
            remote_resource_ctl_port: str = "5000",
            remote_workload_ctl_port: str = "3000",
            target_docker_ctrl_port: str = "4000",
            target_power_broadcaster_port: str = "6000",
            debug = False
        ):
        self.debug = debug
        # LC information
        self.lc_workload_name = lc_workload_name
        self.lc_qos_msec = lc_qos_msec
        self.lc_qos_metric = lc_qos_metric
        self.num_system_gpus = num_system_gpus
        self.lc_max_rps_per_gpu = lc_max_rps_per_gpu
        self.lc_power_profile_dir_path = lc_power_profile_dir_path
        self.lc_load_pct_trace_file = lc_load_pct_trace_file
        self.lc_load_pct_time_unit_sec = lc_load_pct_time_unit_sec
        
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
        
        self.workload_controller = FRSingleGPUWorkloadController(
            power_model=self.power_model.get_power_models(),
            lc_workload_name=lc_workload_name,
            be_workload_name=be_workload_name,
            remote_ip=remote_ip,
            target_ip=target_ip,
            target_docker_ctrl_port=target_docker_ctrl_port,
            remote_resource_ctl_port=remote_resource_ctl_port,
            remote_workload_ctl_port=remote_workload_ctl_port,
            debug=debug
        )

        self.rs_sampler = RSSampler(
            rs_file_path=regulation_signal_file
        )
    def debug_controllers(self):
        assert self.workload_controller.setup_remote(lc_avg_load_pct=50)
        prev = 0
        import random 
        random.seed(110)
        for i in range(100):
            next_ = random.randint(-40,50)
            if prev < 50:
                next_  = abs(next_)
            self.workload_controller.adjust_resources_cap_only(
                    next_power=prev+next_,
                    current_power=prev
                )
            
            prev += next_ + abs(next_//10)
        self.workload_controller.plot_stack()

    def start(
            self, 
            electricity_cost: float,
            regulation_up_reward: float,
            regulation_down_reward: float
        ):
        # Only debug the flow
        if self.debug:
            self.debug_controllers()
            return
        
        lc_load_pct = 10
        optimization_res = self.power_model.optimized_for_fr(
            elec_cost=electricity_cost,
            reg_down=regulation_down_reward,
            reg_up=regulation_up_reward,
            symmetric_provision_range=True
        )
        # if not participating and not in debug mode ...
        if optimization_res['powers']['offset_power'] is 0:
            optimization_res['metrics'] = {
                'perf_score': None,
                'lc_qos_pass': True,
                'be_tp': 1.0
            }
            return optimization_res
        
        print(f"Running frequency regulation for elec price:{electricity_cost} reg_up_reward:{regulation_up_reward} reg_down_reward:{regulation_down_reward}.")
        avg_power = optimization_res['powers']['fr_power']
        max_decrease_watt = optimization_res["regulation"]['reg_up']
        max_increase_watt = optimization_res["regulation"]['reg_down']
        baseline_avg_power = optimization_res['powers']['baseline_power']
        print(f"Baseline avg power: {baseline_avg_power} fr_avg_power: {avg_power} reg_up: {max_decrease_watt} reg_down:{max_increase_watt}.")
        num_chunks = 4
        for chunk_ind in range(num_chunks):
            assert self.workload_controller.setup_remote(lc_load_pct=lc_load_pct)
            if self.debug is False:
                self.start_power_collection()
            print(f"Running LC load from romote trace: {self.lc_load_pct_trace_file}", flush=True)
            #TODO make lc trace chunked too!
            self.workload_controller.run_lc(lc_trace=self.lc_load_pct_trace_file)
            ind = 0
            for rs_val in self.rs_sampler.get_chunk(num_chunks=num_chunks, chunk_ind=chunk_ind)[0]:
                next_power = max_increase_watt if rs_val > 0 else -max_decrease_watt
                next_power *= rs_val
                next_power += avg_power
                curr_power = self.get_current_remote_power()
                print(curr_power)
                return
                curr_power = 0
                print(f"[FREQ_REQ] @{ind}: avg: {avg_power} current: {curr_power} next:{next_power} diff:{next_power- curr_power} rs:{rs_val}", flush=True)
                # input()
                self.workload_controller.adjust_resources_cap_only(
                    next_power=next_power,
                    current_power=curr_power,
                    lc_load_pct=lc_load_pct
                )
                input()
            time.sleep(2)

    def start_power_collection(self):
        assert self.power_collector is None, "[FrequencyRegulator/start_power_collection]: power collector is alreary running"
        self.power_collector = PowerCollector(
            power_broadcaster_ip=self.target_ip,
            power_broadcaster_port=self.target_power_broadcaster_port,
            collection_interval_sec=1,
            debug=self.debug
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
        regulation_signal_file="data/reg_sig_highreg",
        num_system_gpus=8,
        lc_workload_name="Inference-Server",
        lc_qos_msec=40,
        lc_qos_metric="p95th",
        lc_max_rps_per_gpu=95,
        lc_power_profile_dir_path="/home/ajaha004/repos/ROCm-docker-with-controller/results",
        lc_load_pct_trace_file="data/trace_step5_max35.txt",
        lc_load_pct_time_unit_sec=60,
        be_workload_name="miniMDock",
        be_power_profile_dir_path="/home/ajaha004/repos/ROCm-docker-with-controller/results",
        debug=True
    )
    lmps_file = '/home/ajaha004/repos/ROCm-docker-with-controller/scripts/frequency_regulator_power_cap/data/lmpE.npy'
    up_reg_file = '/home/ajaha004/repos/ROCm-docker-with-controller/scripts/frequency_regulator_power_cap/data/lmpRU.npy'
    down_reg_file = '/home/ajaha004/repos/ROCm-docker-with-controller/scripts/frequency_regulator_power_cap/data/lmpRD.npy'
    import numpy as np
    lmps = np.load(lmps_file)
    up_reglmps = np.load(up_reg_file)
    down_reglmps = np.load(down_reg_file)
    for power_price_day, up_price_day, down_price_day in zip(lmps, up_reglmps, down_reglmps):
        day = list()
        for power_price_hr, up_price_hr, down_price_hr in zip(power_price_day, up_price_day, down_price_day):
            
            fr.start(
                electricity_cost=power_price_hr,
                regulation_up_reward=up_price_hr,
                regulation_down_reward=down_price_hr
            )
            return


if __name__ == "__main__":
    test_frequency_regulator()