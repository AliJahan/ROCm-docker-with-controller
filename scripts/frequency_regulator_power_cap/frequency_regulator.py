import os
import sys
import time

scripts_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root_path = os.path.dirname(scripts_path)
sys.path.insert(1, scripts_path) # for utils TODO: make it a wheel
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
            symmetric_provision_range = False,
            print_debug_info = False,
            simulate = False
        ):
        self.simulate = simulate
        self.print_debug_info = print_debug_info
        self.regulation_signal_file = regulation_signal_file
        self.symmetric_provision_range = symmetric_provision_range
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
            lc_load_pct_list=self.power_model.get_lc_load_pct_list(),
            lc_trace_file=lc_load_pct_trace_file,
            lc_workload_name=lc_workload_name,
            be_workload_name=be_workload_name,
            remote_ip=remote_ip,
            target_ip=target_ip,
            target_docker_ctrl_port=target_docker_ctrl_port,
            remote_resource_ctl_port=remote_resource_ctl_port,
            remote_workload_ctl_port=remote_workload_ctl_port,
            num_system_gpus=num_system_gpus,
            print_debug_info=print_debug_info,
            simulate=simulate
        )

        self.rs_sampler = RSSampler(
            rs_file_path=regulation_signal_file,
            keep_symmetric=symmetric_provision_range,
            print_debug_info=print_debug_info
        )

    def debug_controller(self, results_dir: str = "."):
        if self.print_debug_info:
            print(f"[FrequencyRegulator/debug_controller]: simulating power controll", flush=True)
        assert self.workload_controller.setup_remote()
        prev = 0
        import random 
        random.seed(110)
        for chunk_ind in range(4):
            chunk_plot_name = f"chunk{chunk_ind}"
            for i in range(100):
                next_ = random.randint(-40,50)
                if prev < 50:
                    next_  = abs(next_)
                self.workload_controller.adjust_resources_cap_only(
                        next_power=prev+next_,
                        current_power=prev
                    )
                # input("?")
                prev += next_ + abs(next_//10)
            self.workload_controller.plot_states(post_fix=f"{chunk_plot_name}", save_path=f"{results_dir}")
    def create_dirs(self):
        trace_name = self.lc_load_pct_trace_file[self.lc_load_pct_trace_file.rfind("/")+1 : self.lc_load_pct_trace_file.rfind(".")]
        reg_sig_name = self.regulation_signal_file[self.regulation_signal_file.rfind("/")+1:]
        results_dir = f"{project_root_path}/cap_only_gpus{self.num_system_gpus}_trace{trace_name}_reg{reg_sig_name}"
        if self.print_debug_info:
            print(f"[FrequencyRegulator/create_dirs]: creating results dir:{results_dir}", flush=True)
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

    def regulate(
            self, 
            electricity_cost: float,
            regulation_up_reward: float,
            regulation_down_reward: float
        ):
        results_dir = self.create_dirs() # create directories for the results

        # Only debug the flow
        if self.simulate:
            self.debug_controller(results_dir=results_dir)
            return
        
        optimization_res = self.power_model.optimized_for_fr(
            elec_cost=electricity_cost,
            reg_down=regulation_down_reward,
            reg_up=regulation_up_reward,
            symmetric_provision_range=self.symmetric_provision_range
        )
        # if not participating and not in debug mode ...
        if optimization_res['regulation']['reg_up'] is 0 and optimization_res['regulation']['reg_down'] is 0:
            optimization_res['metrics'] = {
                'perf_score': None,
                'lc_qos_pass': True,
                'be_tp': 1.0
            }
            return optimization_res
        
        if self.print_debug_info:
            print(f"[FrequencyRegulator/regulate]: Running frequency regulation for elec price:{electricity_cost} reg_up_reward:{regulation_up_reward} reg_down_reward:{regulation_down_reward}.")
        avg_power = optimization_res['powers']['fr_power']
        max_decrease_watt = optimization_res["regulation"]['reg_up']
        max_increase_watt = optimization_res["regulation"]['reg_down']
        baseline_avg_power = optimization_res['powers']['baseline_power']
        if self.print_debug_info:
            print(f"[FrequencyRegulator/regulate]: Baseline avg power: {baseline_avg_power} fr_avg_power: {avg_power} reg_up: {max_decrease_watt} reg_down:{max_increase_watt}.")
        num_chunks = 15 # 1800/15 = 120 each 2 seconds = 240 sec. Since traces are 15 pcts -> 240/16 = 15 sec each pct
        for chunk_ind in range(num_chunks):
            chunk_plot_name = f"chunk{chunk_ind}"
            if os.path.isfile(results_dir+"/"+chunk_plot_name+".png"):
                if self.print_debug_info:
                    print(f"[FrequencyRegulator/regulate]: Skipping chunk {chunk_ind}", flush=True)
                continue
            assert self.workload_controller.setup_remote()
            self.start_power_collection()
            if self.print_debug_info:
                    print(f"[FrequencyRegulator/regulate]: Running LC load from romote trace: {self.lc_load_pct_trace_file}", flush=True)
            #TODO make lc trace chunked too! now manual based on num_chunk
            self.workload_controller.run_lc(unit_sec=15)
            ind = 0
            for rs_val in self.rs_sampler.get_chunk(num_chunks=num_chunks, chunk_ind=chunk_ind)[0]:
                ind += 1
                next_power = 0
                if self.symmetric_provision_range:
                    next_power = max_increase_watt if rs_val > 0 else -max_decrease_watt
                else:
                    next_power = max_increase_watt
                next_power *= rs_val
                next_power += avg_power
                curr_power = self.get_current_remote_power()['gpu']['total'] - 8*16

                if self.print_debug_info:
                    print(f"[FrequencyRegulator/regulate]: @{ind}: avg: {avg_power} current: {curr_power} next:{next_power} diff:{next_power- curr_power} rs:{rs_val}", flush=True)

                elapsed = self.workload_controller.adjust_resources_cap_only(
                    next_power=next_power,
                    current_power=curr_power
                )
                sleep_time = max(0, 2.0-elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.stop_power_collection()

            if self.print_debug_info:
                    print(f"[FrequencyRegulator/regulate]: plotting states (file:{results_dir}/{chunk_plot_name}.png)",flush=True)
            self.workload_controller.plot_states(post_fix=f"{chunk_plot_name}", save_path=f"{results_dir}")

            trace_save_file = f"{results_dir}/lc_qos"
            if self.print_debug_info:
                    print(f"[FrequencyRegulator/regulate]: saving lc qos data (file:{trace_save_file})", flush=True)
            f = open(f"{trace_save_file}", 'a+')
            f.write(f"chunk {chunk_ind}: "+str(self.workload_controller.get_lc_results())+"\n")
            f.close()

            if self.print_debug_info:
                    print(f"[FrequencyRegulator/regulate]: cleaning up experiment (chunk:{chunk_ind}/{num_chunks})", flush=True)

            self.workload_controller.cleanup_remote_and_save_results(save_path=results_dir, chunk=chunk_ind)


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
        assert self.power_collector is not None, "[FrequencyRegulator/stop_power_collection]: power collector is not running"
        power = self.power_collector.get_all_powers()
        del self.power_collector
        self.power_collector = None
        return power

    def get_current_remote_power(self):
        assert self.power_collector is not None, "[FrequencyRegulator/get_current_remote_power]: power collector is not running"
        return self.power_collector.get_cur_power()

    def __del__(self):
        # self.workload_controller.plot_stack(post_fix="test")
        del self.workload_controller

def test_frequency_regulator():
    
    lmps_file = f'{scripts_path}/frequency_regulator_power_cap/data/lmpE.npy'
    up_reg_file = f'{scripts_path}/frequency_regulator_power_cap/data/lmpRU.npy'
    down_reg_file = f'{scripts_path}/frequency_regulator_power_cap/data/lmpRD.npy'
    import numpy as np
    lmps = np.load(lmps_file)
    up_reglmps = np.load(up_reg_file)
    down_reglmps = np.load(down_reg_file)
    # for power_price_day, up_price_day, down_price_day in zip(lmps, up_reglmps, down_reglmps):
    #     day = list()
    #     for power_price_hr, up_price_hr, down_price_hr in zip(power_price_day, up_price_day, down_price_day):
    for reg_sing in ['reg_sig_midreg1', 'reg_sig_midreg2', 'reg_sig_highreg']:
         for lc_trace in ['trace_step5_max60.txt', 'trace_step5_max80.txt', 'trace_step5_max40.txt', 'trace_fb_1_1.txt', 'trace_fb_1_2.txt', 'trace_fb_2_1.txt', 'trace_fb_2_2.txt']:
            fr = FrequencyRegulator(
                regulation_signal_file=f"{scripts_path}/frequency_regulator_power_cap/data/{reg_sing}",
                num_system_gpus=7,
                lc_workload_name="Inference-Server",
                lc_qos_msec=40,
                lc_qos_metric="p95th",
                lc_max_rps_per_gpu=95,
                lc_power_profile_dir_path=f"{scripts_path}/../results",
                lc_load_pct_trace_file=f"{scripts_path}/frequency_regulator_power_cap/data/{lc_trace}",
                lc_load_pct_time_unit_sec=60,
                be_workload_name="miniMDock",
                be_power_profile_dir_path=f"{scripts_path}/../results",
                symmetric_provision_range=False,
                print_debug_info=True,
                simulate=True
            )
            res = fr.regulate(
                electricity_cost=100,
                regulation_up_reward=0,
                regulation_down_reward=10
            )
            # if 
            del fr

            input("?")
    print("Done with experiments")
    return

if __name__ == "__main__":
    test_frequency_regulator()