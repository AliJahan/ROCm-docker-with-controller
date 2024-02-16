import os
import pandas as pd
from sklearn import linear_model

from fr_optimizer import Optimize

class PowerModel:
    def __init__(
            self,
            lc_qos_msec: int,
            lc_qos_metric: str,
            num_system_gpus: int,
            lc_max_rps_per_gpu: int,
            lc_power_profile_dir_path: str,
            lc_load_pct_trace_file: str,
            be_power_profile_dir_path: str,
        ):
        self.lc_qos_msec = lc_qos_msec
        self.lc_qos_metric = lc_qos_metric
        self.num_system_gpus = num_system_gpus
        self.lc_max_rps_per_gpu = lc_max_rps_per_gpu
        self.lc_load_pct_trace_file = lc_load_pct_trace_file
        self.lc_power_profile_dir_path = lc_power_profile_dir_path
        self.be_power_profile_dir_path = be_power_profile_dir_path
        self.power_profiles = self.load_power_profiles()
        self.lc_load_trace = self.load_and_convert_lc_load_trace()
        self.power_models = self.create_power_model(plot=True)
        # self.lc_trace_powers = self.get_power_ranges_lc_cap()

    def load_lc_power_profile(self):
        # Obtain all profiled files (CUMasking + PowerCapping)
        lc_powercap_files = list()
        lc_cu_files = list()
        for (dirpath, _, filenames) in os.walk(self.lc_power_profile_dir_path):
            # print(f"dirpath: {dirpath} dirname: {dirnames} filenames: {filenames}")
            for _file in filenames:
                if "cap" in _file: # power cap profile
                    lc_powercap_files.append(f"{dirpath}/{_file}")
                elif "cus" in _file: # cus profile
                    lc_cu_files.append(f"{dirpath}/{_file}")
                else:
                    print(f"[FrequencyRegulator/load_lc_power_profile]: profile file name not supported: {_file}")
        # Process PowerCapping profile data
        # sort based on cap value
        lc_powercap_files.sort(
            key=lambda file_abs_path: 
                int(
                    (
                        file_abs_path[file_abs_path.rfind('/')+1:]
                    ).split("cap")[1].split("_")[0]
                )
        )
        lc_cap_power_map = dict()
        for file_abs_path in lc_powercap_files:
            file_name = file_abs_path[file_abs_path.rfind('/')+1:]
            cap = int(file_name.split("cap")[1].split("_")[0])
            print(f"[PowerModel/load_lc_power_profile]: Reading file {file_name}, PowerCap {cap}")
            all_data = pd.read_csv(file_abs_path)
            all_data = all_data.sort_values('load_pct')
            filtered_by_qos = all_data[all_data[self.lc_qos_metric] <= self.lc_qos_msec]
            lc_cap_power_map[cap] = filtered_by_qos

            # input(pd.DataFrame(filtered_by_qos, columns=['load_pct', 'load_rps', 'gpu_0_avg_pow', self.lc_qos_metric]))
        # Process CUMasking profile data
        # sort based on cu value
        lc_cu_files.sort(
            key=lambda file_abs_path: 
                int(
                    (
                        file_abs_path[file_abs_path.rfind('/')+1:]
                    ).split("cus")[1].split("_")[0]
                )
        )
        lc_cu_power_map = dict()
        for file_abs_path in lc_cu_files:
            file_name = file_abs_path[file_abs_path.rfind('/')+1:]
            cus = int(file_name.split("cus")[1].split("_")[0])
            print(f"[PowerModel/load_lc_power_profile]: Reading file {file_name}, CUs {cus}")
            all_data = pd.read_csv(file_abs_path)
            all_data = all_data.sort_values('load_pct')
            filtered_by_qos = all_data[all_data[self.lc_qos_metric] <= self.lc_qos_msec]
            lc_cu_power_map[cus] = filtered_by_qos
            
        
        return {
            'cap': lc_cap_power_map, 
            'cu': lc_cu_power_map
        }

    def load_and_convert_lc_load_trace(self):
        """
        Loads LC load trace. Each line of the trace has to be between 1 to 100.
        Then the load gets converted to number of GPUs needed based on 
        `lc_max_rps_per_gpu` and `num_system_gpus`.

        Returns:
            dict[str]->list: 'raw_percent', 'single_gpu_rps', 'server_rps', 'num_gpus'
        """          
        # Load data from given load trace 
        f = open(self.lc_load_pct_trace_file, 'r')
        lines = f.readlines()
        f.close()
        
        # Process
        load_pct_data = list()
        load_server_rps_data = list()
        load_single_gpu_rps_data = list()
        load_num_gpu_data = list()
        for line in lines:
            if len(line) == 1: # empty lines
                continue
            assert float(line) >= 1.0 and float(line) <= 100.0, f"[PowerModel/load_lc_load_trace]: Error in loading LC load trace {line} is not within range [1,100]"
            load_pct = float(line)
            load_pct_data.append(load_pct)
            num_gpus = (load_pct/100.0) * self.num_system_gpus
            server_rps = num_gpus * self.lc_max_rps_per_gpu
            single_gpu_rps = (load_pct/100.0) * self.lc_max_rps_per_gpu
            load_single_gpu_rps_data.append(single_gpu_rps)
            load_num_gpu_data.append(num_gpus)
            load_server_rps_data.append(server_rps)

        return {
            'raw_percent': load_pct_data,
            'single_gpu_rps': load_single_gpu_rps_data,
            'server_rps': load_server_rps_data,
            'num_gpus': load_num_gpu_data
        }

    def load_be_power_profile(self):
        pass
    
    def load_power_profiles(self):
        lc_power_profile = self.load_lc_power_profile()
        be_power_profile = self.load_be_power_profile()
        return {
            'be' : be_power_profile,
            'lc' : lc_power_profile
        }
    
    def create_power_model(self, plot = True, plot_save_path: str = "../data"):
        if plot:
            import matplotlib.pyplot as plt
        # Create power model 
        power_profile = self.power_profiles
        power_model = dict()
        for workload in power_profile:
            power_model[workload] = dict()
            if power_profile[workload] is None: # BE is not ready yet
                continue
            for model_type in power_profile[workload]: # cu or cap
                power_model[workload][model_type] = dict()
                for type_value in power_profile[workload][model_type]: # num_cus or power_cap value
                    powers = power_profile[workload][model_type][type_value].gpu_0_avg_pow.values
                    powers = powers.reshape(powers.shape[0], 1)
                    loads_pct = power_profile[workload][model_type][type_value].load_pct.values
                    loads_pct = loads_pct.reshape(loads_pct.shape[0], 1)
                    regressor = linear_model.LinearRegression()
                    regressor.fit(loads_pct, powers)
                    power_model[workload][model_type][type_value] = regressor.predict
                    if plot:
                        plt.cla()
                        plt.scatter(loads_pct, powers,  color='black', label="Power Data")
                        # plt.xlim(min(loads_pct)-10,max(loads_pct)+10)
                        plt.plot(loads_pct, regressor.predict(loads_pct), color='blue', linewidth=3, label="Power Model")
                        coef = "{:.2f}".format(regressor.coef_[0][0])
                        intercept = "{:.2f}".format(regressor.intercept_[0])
                        label_y_coord = max(regressor.predict(loads_pct))
                        info_str = f"qos metric: {self.lc_qos_metric}\nqos threashold:{self.lc_qos_msec} msec\nmax safe load: {max(loads_pct)[0]}\n------\ncoef: {coef}\nintercept: {intercept}"
                        plt.text(0,160, info_str)
                        plt.title(f"Power/Load for {model_type} = {type_value}")
                        plt.xlabel("Load (%)")
                        plt.ylim(0,225)
                        plt.ylabel("Avg. Power (w)")
                        plt.legend(loc='upper right')
                        plt.savefig(f"{plot_save_path}/lc_load_vs_{model_type}{type_value}_regression_model.png")
        return power_model

    def get_power_ranges_lc_cap(self, avg_load_pct):
        # Compute avg. power for LC
        lc_power_model = self.power_models['lc']
        # Getting minimum power cap for max lc load
        # lc_loads_max = max(self.lc_load_trace['raw_percent'])
        lowest_safe_power_cap = 225
        for i in self.power_profiles['lc']['cap']:
            df = self.power_profiles['lc']['cap'][i]
            if df[df.load_pct > avg_load_pct].shape[0] > 0:
                lowest_safe_power_cap = min(lowest_safe_power_cap, i)
                break
        # Convert to proper shape (2D)
        lc_trace_loads = [[l] for l in self.lc_load_trace['raw_percent']]
        
        # Map to lowest safe power using regression model
        # print(f"{lowest_safe_power_cap}")
        lc_lowest_safe_powers = lc_power_model['cap'][lowest_safe_power_cap](lc_trace_loads)
        lc_lowest_safe_powers = [i[0] for i in lc_lowest_safe_powers]
        # Map to highest safe power using regression model
        lc_highest_safe_powers = lc_power_model['cap'][225](lc_trace_loads)
        lc_highest_safe_powers = [i[0] for i in lc_highest_safe_powers]
        # print(lc_lowest_safe_powers)
        # print(lc_highest_safe_powers)
        load_powers_dict = {
            'load_pct': self.lc_load_trace['raw_percent'],
            'lowest_safe_power': lc_lowest_safe_powers,
            'highest_safe_power': lc_highest_safe_powers
        }
        load_trace_powers = pd.DataFrame(load_powers_dict)
        # print(load_trace_powers.describe())
        return load_trace_powers

    def optimized_for_fr(self, elec_cost: int, reg_up: int, reg_down: int, symmetric_provision_range=False, preformance_score_pct=85):
        avg_load = sum(self.lc_load_trace['raw_percent'])/len(self.lc_load_trace['raw_percent'])
        power_ranges = self.get_power_ranges_lc_cap(avg_load)
        # print(power_ranges.describe())
        min_acceptable_power = self.num_system_gpus*power_ranges.lowest_safe_power.mean()
        avg_power = self.num_system_gpus*power_ranges.highest_safe_power.mean()
        # print(min_acceptable_power)
        # print(avg_power)
        res = Optimize(
            avg_power=avg_power,
            min_acceptable_power=min_acceptable_power,
            increase_price=reg_down,
            max_power=self.num_system_gpus*220,
            reduce_price=reg_up,
            power_price=elec_cost,
            saving_threashold_pct=20,
            symmetric_provision_range=symmetric_provision_range,
            predicted_perf_score_pct=preformance_score_pct
        )
        if res is None:
            return {
                "costs": {
                    "baseline_elec_cost": avg_power * elec_cost,
                    "fr_elec_cost": 0,
                    'reg_up_reward': 0,
                    'reg_down_reward': 0,
                    'fr_total_cost': avg_power * elec_cost,
                    'saving': 0
                },
                "regulation": {
                    'reg_up': 0,
                    'reg_down': 0
                },
                "powers": {
                    'baseline_power': avg_power,
                    'offset_power': 0,
                    'fr_power': avg_power
                }
            }
        return res
            



def test_power_model():
    pm = PowerModel(
        num_system_gpus=8,
        lc_qos_msec=40,
        lc_qos_metric="p99th",
        lc_max_rps_per_gpu=98,
        lc_load_pct_trace_file="/home/ajaha004/repos/ROCm-docker-with-controller/scripts/frequency_regulator_power_cap/data/example_lc_trace.txt",
        be_power_profile_dir_path="/home/ajaha004/repos/ROCm-docker-with-controller/results",
        lc_power_profile_dir_path="/home/ajaha004/repos/ROCm-docker-with-controller/results"
    )
    return
    lmps_file = '/home/ajaha004/repos/ROCm-docker-with-controller/scripts/frequency_regulator_power_cap/data/lmpE.npy'
    up_reg_file = '/home/ajaha004/repos/ROCm-docker-with-controller/scripts/frequency_regulator_power_cap/data/lmpRU.npy'
    down_reg_file = '/home/ajaha004/repos/ROCm-docker-with-controller/scripts/frequency_regulator_power_cap/data/lmpRD.npy'
    import numpy as np
    results = list()
    lmps = np.load(lmps_file)
    up_reglmps = np.load(up_reg_file)
    down_reglmps = np.load(down_reg_file)
    non_zero = 0
    all_hours = 0
    norm_saving = list()
    all_norm_saving = list()
    for power_price_day, up_price_day, down_price_day in zip(lmps, up_reglmps, down_reglmps):
        day = list()
        for power_price_hr, up_price_hr, down_price_hr in zip(power_price_day, up_price_day, down_price_day):
            res = pm.optimized_for_fr(
                elec_cost=power_price_hr,
                reg_down=down_price_hr,
                reg_up=up_price_hr,
                symmetric_provision_range=False
                )
            day.append((power_price_hr, up_price_hr, down_price_hr, res['powers']['baseline_power'], res['powers']['fr_power'], res['regulation']['reg_up'], res['regulation']['reg_down']))
            all_hours += 1
            all_norm_saving.append(res['costs']['saving']/res['costs']['baseline_elec_cost'])
            if res['regulation']['reg_up']>0 or  res['regulation']['reg_down']>0:
                norm_saving.append(res['costs']['saving']/res['costs']['baseline_elec_cost'])
                # print(f"{res}\n")
                non_zero += 1
        results.append(day)
    print(f"participation percent: {non_zero*100/all_hours} ({non_zero}/{all_hours})")
    print(f"Participation avg normalized saving {sum(norm_saving)/len(norm_saving)}")
    print(f"Overall avg normalized saving {sum(all_norm_saving)/len(all_norm_saving)}")

    # print(res)

if __name__ == "__main__":
    test_power_model()