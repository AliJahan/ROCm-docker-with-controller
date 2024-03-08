import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, ensemble, neural_network

from .fr_optimizer import Optimize
from .plot_be_profile import BEPowerProfilePlotter
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
            debug = False
        ):
        self.debug = debug
        self.lc_qos_msec = lc_qos_msec
        self.lc_qos_metric = lc_qos_metric
        self.num_system_gpus = num_system_gpus
        self.lc_max_rps_per_gpu = lc_max_rps_per_gpu
        self.lc_load_pct_trace_file = lc_load_pct_trace_file
        self.lc_power_profile_dir_path = lc_power_profile_dir_path
        self.be_power_profile_dir_path = be_power_profile_dir_path
        self.power_profiles = self.load_power_profiles()
        self.lc_load_trace = self.load_and_convert_lc_load_trace()
        self.power_models = self.create_power_model()
        # self.lc_trace_powers = self.get_power_ranges_lc_cap()

    def get_lc_load_pct_list(self):
        return self.lc_load_trace['raw_percent']

    def get_power_models(self):
        return self.power_models

    def load_power_profiles(self):
        # Obtain all profiled files (CUMasking + PowerCapping)
        lc_powercap_files = list()
        lc_cu_files = list()
        be_cu_cap_files = list()
        for (dirpath, _, filenames) in os.walk(self.lc_power_profile_dir_path):
            for _file in filenames:
                if "lc" in _file and "_cap" in _file: # power cap profile
                    lc_powercap_files.append(f"{dirpath}/{_file}")
                elif "lc" in _file and "_cus" in _file: # cus profile
                    lc_cu_files.append(f"{dirpath}/{_file}")
                elif "be" in _file:
                    be_cu_cap_files.append(f"{dirpath}/{_file}")
                else:
                    print(f"[FrequencyRegulator/load_power_profiles]: profile file name not supported: {_file}")
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
            if self.debug:
                print(f"[PowerModel/load_power_profiles]: Reading file {file_name}, PowerCap {cap}")
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
            if self.debug:
                print(f"[PowerModel/load_power_profiles]: Reading file {file_name}, CUs {cus}")
            all_data = pd.read_csv(file_abs_path)
            all_data = all_data.sort_values('load_pct')
            filtered_by_qos = all_data[all_data[self.lc_qos_metric] <= self.lc_qos_msec]
            lc_cu_power_map[cus] = filtered_by_qos

        assert len(be_cu_cap_files) == 1, f"[PowerModel/load_power_profiles]: there are multiple be profiles: {be_cu_cap_files}"
        
        be_power_df = None
        be_cap_cu_power_map = dict()
        for file_abs_path in be_cu_cap_files:
            file_name = file_abs_path[file_abs_path.rfind('/')+1:]
            if self.debug:
                print(f"[PowerModel/load_be_power_profile]: Reading file {file_name}")
            all_data = pd.read_csv(file_abs_path)
            all_data = all_data.sort_values(['cap', 'cu'], ascending=[False, False])
            # TODO: eyeballing data shows cu=12 is supported by all of cap,cu combinations
            filtered_idle = all_data[all_data.gpu > 0]
            filtered_by_cu = filtered_idle[filtered_idle.cu >= 12]
            filtered_by_gpu_idle_power = filtered_by_cu[filtered_by_cu.cap >= 10] # gpu idle power
            be_power_df = filtered_by_gpu_idle_power
            for idx, row in be_power_df.iterrows():
                if row.cap not in be_cap_cu_power_map:
                    be_cap_cu_power_map[row.cap] = dict()
                be_cap_cu_power_map[row.cap][row.cu] = row.gpu_0_avg_pow
        if self.debug:
            plotter = BEPowerProfilePlotter(be_power_df, f"{os.path.dirname(os.path.abspath(__file__))}/../data/", "miniMDock")
            plotter.plot()
        
        return {
            'lc': {
                'cap': lc_cap_power_map, 
                'cu': lc_cu_power_map
            },
            'be': be_power_df
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
    
    def create_lc_power_model(self, power_profile, plot_save_path: str):
        power_model = dict()
        for model_type in power_profile: # cu or cap
                if self.debug:
                    plt.cla()
                power_model[model_type] = dict()
                for type_value in power_profile[model_type]: # num_cus or power_cap value
                    powers = power_profile[model_type][type_value].gpu_0_avg_pow.values
                    powers = powers.reshape(powers.shape[0], 1)
                    loads_pct = power_profile[model_type][type_value].load_pct.values
                    loads_pct_extended = loads_pct.reshape(loads_pct.shape[0], 1)
                    regressor = linear_model.LinearRegression()
                    regressor.fit(loads_pct_extended, powers)
                    power_model[model_type][type_value] = regressor.predict
                    if self.debug:
                        # plt.cla()
                        # coef = "{:.2f}".format(regressor.coef_[0][0])
                        # intercept = "{:.2f}".format(regressor.intercept_[0])
                        # info_str = f"qos metric: {self.lc_qos_metric}\nqos threashold:{self.lc_qos_msec} msec\nmax safe load: {max(loads_pct)[0]}\n------\ncoef: {coef}\nintercept: {intercept}"
                        # plt.text(0,160, info_str)
                        # plt.scatter(loads_pct, powers,  color='black', label="Power Data")
                        plt.plot(loads_pct, regressor.predict(loads_pct_extended), linewidth=3, label=f"{model_type}={type_value}")                        
                        plt.title(f"Power/Load for {model_type} = {type_value}")
                        plt.xlabel("Load (%)")
                        plt.ylim(0,225)
                        plt.xlim(-5,105)
                        plt.ylabel("Avg. Power (w)")
                        plt.legend(loc='upper right',ncol=4, bbox_to_anchor=(1, 1))
                        plt.savefig(f"{plot_save_path}/lc_load_vs_{model_type}{type_value}.png", tight_layout=True)
        return power_model
    
    def create_be_power_model(self, power_profile, plot_save_path: str):
        power_model = {
            'power2cap': dict(),
            'power2cu': dict()
        }
        if self.debug:
            plt.cla()
        
        caps = sorted(list(power_profile.cap.unique()))
        cus = sorted(list(power_profile.cu.unique()))
        for cu in cus:
            filtered_by_cu = power_profile.query(f"cu == {cu}")
            # filter caps that do not work (the point where the actual power is always less than cap value)
            filtered_max_power = filtered_by_cu.query(f'cap <= {max(filtered_by_cu.gpu_0_avg_pow.values)}')
            # filter caps that do not work (the point where the actual power is always greater than cap value)
            filtered_power = filtered_max_power.query(f'cap > {min(filtered_max_power.gpu_0_avg_pow.values)}')
            avg_powers = filtered_power.gpu_0_avg_pow.values
            avg_powers_extended = avg_powers.reshape(avg_powers.shape[0], 1)
            caps = filtered_power.cap.values
            caps_extended = caps.reshape(caps.shape[0], 1)
            # regressor = linear_model.LinearRegression()
            # regressor.fit(avg_powers_extended, caps)
            power_model['power2cap'][cu] = {
                # 'reg_model': regressor.predict,
                'min_supported': min(avg_powers),
                'max_supported': max(avg_powers)
            }
            
            if self.debug:
                plt.cla()
                # coef = "{:.2f}".format(regressor.coef_[0][0])
                # intercept = "{:.2f}".format(regressor.intercept_[0])
                # info_str = f"qos metric: {self.lc_qos_metric}\nqos threashold:{self.lc_qos_msec} msec\nmax safe load: {max(loads_pct)[0]}\n------\ncoef: {coef}\nintercept: {intercept}"
                # plt.text(0,160, info_str)
                plt.scatter(avg_powers, caps,  color='black', label="Power Data")
                # plt.plot(avg_powers, regressor.predict(caps_extended), linewidth=3, label=f"cu={cu}")                        
                plt.title(f"Power/Load for cu = {cu}")
                plt.xlabel("Power")
                # plt.ylim(0,225)
                # plt.xlim(-5,105)
                plt.ylabel("Cap")
                plt.legend(loc='upper right',ncol=4, bbox_to_anchor=(1, 1))
                plt.savefig(f"{plot_save_path}/be_power2cap_cu{cu}.png", tight_layout=True)
        min_power_by_cap = max(power_profile.query(f"cap == {power_profile.cap.min()}").gpu_0_avg_pow.values)
        # print(min_power_by_cap)
        # input("?")
        for cap in caps:
            filtered_by_cap = power_profile.query(f"cap == {cap}")
            if cap < min_power_by_cap:
                # filter cus that do not work (the point where the actual power is always less than cap value)
                filtered_by_cap = filtered_by_cap.query(f'gpu_0_avg_pow >= {cap}')
                filtered_by_cap = filtered_by_cap.query(f'cap <= {min_power_by_cap}')

            # filter caps that do not work (the point where the actual power is always greater than cap value)
            # filtered_cu = filtered_max_cu.query(f'cap > {min(filtered_max_cu.gpu_0_avg_pow.values)}')
            avg_powers = filtered_by_cap.gpu_0_avg_pow.values
            avg_powers_extended = avg_powers.reshape(avg_powers.shape[0], 1)
            cus = filtered_by_cap.cu.values
            cus_extended = cus.reshape(cus.shape[0], 1)

            # regressor = neural_network.MLPRegressor(hidden_layer_sizes=(100,100))
            # regressor = linear_model.LinearRegression()
            # regressor.fit(avg_powers_extended, cus)
            power_model['power2cu'][cap] = {
                # 'reg_model': regressor.predict,
                'min_supported': min(avg_powers),
                'max_supported': max(avg_powers)
            }

            if self.debug:
                plt.cla()
                # coef = "{:.2f}".format(regressor.coef_[0][0])
                # intercept = "{:.2f}".format(regressor.intercept_[0])
                # info_str = f"qos metric: {self.lc_qos_metric}\nqos threashold:{self.lc_qos_msec} msec\nmax safe load: {max(loads_pct)[0]}\n------\ncoef: {coef}\nintercept: {intercept}"
                # plt.text(0,160, info_str)
                # plt.scatter(cus, avg_powers, color='black', label="Power Data")
                plt.scatter(avg_powers, cus, color='black', label="Power Data")
                # plt.plot(avg_powers, regressor.predict(cus_extended), linewidth=3, label=f"cap={cap}")                        
                plt.title(f"Power/Load for cap = {cap}")
                plt.xlabel("Power")
                # plt.ylim(0,225)
                # plt.x               
                plt.ylabel("CU")
                plt.legend(loc='upper right',ncol=4, bbox_to_anchor=(1, 1))
                plt.savefig(f"{plot_save_path}/be_power2cu_cap{cap}.png", tight_layout=True)
            # input("?")
        return power_model
    
    def create_power_model(self, plot_save_path: str = f"{os.path.dirname(os.path.abspath(__file__))}/../data"):
        
        power_model = {
            'lc': self.create_lc_power_model(
                    power_profile=self.power_profiles['lc'],
                    plot_save_path=plot_save_path
                ),
            'be': self.create_be_power_model(
                    power_profile=self.power_profiles['be'], 
                    plot_save_path=plot_save_path
                )
        }
                
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

    def optimized_for_fr(self, elec_cost: int, reg_up: int, reg_down: int, symmetric_provision_range=False, preformance_score_pct=95):
        avg_load = sum(self.lc_load_trace['raw_percent'])/len(self.lc_load_trace['raw_percent'])
        power_ranges = self.get_power_ranges_lc_cap(avg_load)
        # print(power_ranges.describe())
        min_acceptable_power = self.num_system_gpus*power_ranges.lowest_safe_power.mean()
        avg_power = self.num_system_gpus*power_ranges.highest_safe_power.mean()
        res = Optimize(
            avg_power=avg_power,
            min_acceptable_power=avg_power,
            increase_price=reg_down,
            max_power=self.num_system_gpus*self.power_models['be']['power2cap'][60]['max_supported'],
            reduce_price=reg_up,
            power_price=elec_cost,
            saving_threashold_pct=10,
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
        lc_qos_metric="p95th",
        lc_max_rps_per_gpu=98,
        lc_load_pct_trace_file="/home/ajaha004/repos/ROCm-docker-with-controller/scripts/frequency_regulator_power_cap/data/trace_step5_max60.txt",
        be_power_profile_dir_path="/home/ajaha004/repos/ROCm-docker-with-controller/results",
        lc_power_profile_dir_path="/home/ajaha004/repos/ROCm-docker-with-controller/results",
        debug=True
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