import os
import sys
import math
import enum
import copy
import datetime
import pickle
from inspect import currentframe, getframeinfo
import time
sys.path.insert(1, '../../') # for utils TODO: make it a wheel

from utils.stat_collectors.lc_load_collector import LCLoadCollector
from utils.resource_managers.resource_manager_distributed import GPUResourceManagerDistributed, GPUWorkloadType
from utils.workload_runners.workload_runner_remote import RemoteDockerRunner, RemoteWorkloadRunner
from utils.workload_runners.lc_runner.lc_client_runner import LCClientRunner

class WorkloadState(enum.Enum):
    PAUSED = 0
    RUNNING = 1

class FRSingleGPUWorkloadController:
    lc_client_runner = None
    lc_num_gpus_needed = 1
    workloads_state = dict()
    
    def __init__(
            self,
            power_model,
            lc_load_pct_list,
            lc_trace_file: str,
            lc_model_name: str = "resnet152",
            lc_batch_size: int = 1,
            lc_workload_name: str = "Inference-Server",
            be_workload_name: str= "miniMDock",
            target_ip: str = "172.20.0.9",
            remote_ip: str = "172.20.0.6",
            remote_resource_ctl_port: str = "5000",
            remote_workload_ctl_port: str = "3000",
            target_docker_ctrl_port: str = "4000",
            num_system_gpus: int = 8,
            print_debug_info = False,
            simulate = False
        ):
        self.simulate = simulate
        self.print_debug_info=print_debug_info

        self.lc_trace_file = lc_trace_file
        self.num_system_gpus = num_system_gpus
        self.RESULTS_DIR = "/workspace/fr_results"
        self.LOGS_DIR = "/workspace/experiment_logs"
        self.target_ip = target_ip
        self.remote_ip = remote_ip
        self.power_model = power_model
        self.lc_load_pct_list = lc_load_pct_list
        self.lc_model_name = lc_model_name
        self.lc_batch_size = lc_batch_size
        self.lc_workload_name = lc_workload_name
        self.be_workload_name = be_workload_name
        self.remote_resource_ctl_port = remote_resource_ctl_port
        self.remote_workload_ctl_port = remote_workload_ctl_port
        self.target_docker_ctrl_port = target_docker_ctrl_port

        self.docker_runner = RemoteDockerRunner(
            remote_ip=remote_ip,
            target_ip=target_ip,
            target_docker_control_port=target_docker_ctrl_port,
            remote_workload_control_port=remote_workload_ctl_port,
            remote_resource_ctl_port=remote_resource_ctl_port,
            print_debug_info=print_debug_info,
            simulate=simulate
        )

        self.workload_runner = RemoteWorkloadRunner(
            remote_ip=self.remote_ip,
            target_ip=self.target_ip,
            remote_workload_control_port=self.remote_workload_ctl_port,
            lc_workload_name=lc_workload_name,
            be_workload_name=be_workload_name,
            wait_after_send=False,
            print_debug_info=print_debug_info,
            simulate=simulate
        )
        
        self.resource_controller = GPUResourceManagerDistributed(
            remote_control_ip=remote_ip,
            remote_control_port=remote_resource_ctl_port,
            wait_after_send=False,
            print_debug_info=print_debug_info,
            simulate=simulate
        )
        
        for workload in [GPUWorkloadType.BE, GPUWorkloadType.LC]:
            self.workloads_state[workload] = dict()
        self.be_capped_stack = list()
        self.plot_data = list()

    def time_str(self):
        return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    def get_cur_internal_power(self):
        curr_internal_powers = 0
        for gpu in self.be_capped_stack:
            curr_internal_powers += self.workloads_state[GPUWorkloadType.BE][gpu]['cap']['current']
        return curr_internal_powers

    def save_plot_snapshot(self, target_power: int, current_power: int):
        self.plot_data.append(
            {
                'time': self.time_str(),
                'target' : target_power,
                'current': current_power,
                'wl_state': copy.deepcopy(self.workloads_state),
                'capp_stack': copy.deepcopy(self.be_capped_stack)
            }
        )

    def decrease_gpu_power(self, target_power: int, current_power: int):
        amount_watt =  current_power - target_power
        max_supported_power = int(self.power_model['be']['power2cap'][60]['max_supported'])
        min_supported_power = int(self.power_model['be']['power2cap'][60]['min_supported'])

        # Check if with current set of gpus we can reach target_power
        # curr_internal_powers = self.get_cur_internal_power()
        # if curr_internal_powers <= target_power:
        #     if self.debug:
        #         print(f"\t-[FRWorkloadController/decrease:{getframeinfo(currentframe()).lineno}]: skipping! (internal_power:{curr_internal_powers} target:{target_power})", flush=True)

        #     self.save_plot_snapshot(target_power=target_power, current_power=current_power)
        #     return
        # else:
        #     amount_watt =  curr_internal_powers - target_power
        # if self.debug:
        #     print(f"\t-[FRWorkloadController/decrease:{getframeinfo(currentframe()).lineno}]: adjusted! (cur:{current_power} target:{target_power} internal_power:{curr_internal_powers} amount:{amount_watt})", flush=True)
        
        got_in = False
        while amount_watt > 1 and len(self.be_capped_stack) > 0:
            got_in = True
            capped_tos = self.be_capped_stack[-1] # tos: top of stack
            # up room left to increase cap
            capped_tos_power = int(self.workloads_state[GPUWorkloadType.BE][capped_tos]['cap']['current'])
            down_room_left_on_tos = capped_tos_power - min_supported_power
            if self.print_debug_info:
                print(f"\t-[FRWorkloadController/decrease:{getframeinfo(currentframe()).lineno}]: Begin Loop (tos:{capped_tos} cur_tos_power:{capped_tos_power} down_room:{down_room_left_on_tos} amount:{amount_watt})", flush=True)
            # 1
            if amount_watt > capped_tos_power: #  
                (pasued_gpu, res) = self.pause_one_gpu(is_be=True)
                if res:
                    self.workloads_state[GPUWorkloadType.BE][pasued_gpu]['cap']['current'] = max_supported_power
                    self.resource_controller.set_freq(
                        app_name=self.be_workload_name,
                        gpu=pasued_gpu,
                        freq=225
                    )
                    amount_watt -= capped_tos_power
                    if self.print_debug_info:
                        print(f"\t-[FRWorkloadController/decrease:{getframeinfo(currentframe()).lineno}]: gpu {pasued_gpu} paused due to not enough cap room! (reduction by pause:{capped_tos_power} new amount:{amount_watt})", flush=True)
                else:
                    if self.print_debug_info:
                        print(f"\t-[FRWorkloadController/decrease:{getframeinfo(currentframe()).lineno}]: No GPU found to pasue, (amount_watt:{amount_watt})", flush=True)
                    break
            else:
                if down_room_left_on_tos > 0: # first cap to min if any down room left on tos
                    if amount_watt > down_room_left_on_tos:
                        new_cap = min_supported_power
                        self.workloads_state[GPUWorkloadType.BE][capped_tos]['cap']['current'] = new_cap
                        self.resource_controller.set_freq(
                            app_name=self.be_workload_name,
                            gpu=capped_tos,
                            freq=int(new_cap)
                        )
                        amount_watt -= down_room_left_on_tos
                        if self.print_debug_info:
                            print(f"\t-[FRWorkloadController/decrease:{getframeinfo(currentframe()).lineno}]: down room less than amount, capping to min! (gpu:{capped_tos} prev_cap:{capped_tos_power} new_cap:{min_supported_power} downroom:{down_room_left_on_tos} amount:{amount_watt})", flush=True)
                    else:
                        new_cap = capped_tos_power - amount_watt
                        self.workloads_state[GPUWorkloadType.BE][capped_tos]['cap']['current'] = new_cap
                        self.resource_controller.set_freq(
                            app_name=self.be_workload_name,
                            gpu=capped_tos,
                            freq=int(new_cap)
                        )
                        amount_watt = 0
                        if self.print_debug_info:
                            print(f"\t-[FRWorkloadController/decrease:{getframeinfo(currentframe()).lineno}]: down room can match amount, matched! (gpu:{capped_tos} prev_cap:{capped_tos_power} new_cap:{min_supported_power} downroom:{down_room_left_on_tos} amount:{amount_watt})", flush=True)
                elif amount_watt < capped_tos_power - min_supported_power//2:
                    new_cap = min_supported_power
                    self.workloads_state[GPUWorkloadType.BE][capped_tos]['cap']['current'] = new_cap
                    self.resource_controller.set_freq(
                        app_name=self.be_workload_name,
                        gpu=capped_tos,
                        freq=int(new_cap)
                    )
                    amount_watt -= min_supported_power 
                    if self.print_debug_info:
                        print(f"\t-[FRWorkloadController/decrease:{getframeinfo(currentframe()).lineno}]: no downroom but amount less than half of min, (gpu:{capped_tos} prev_cap:{capped_tos_power} new_cap:{new_cap} amount:{amount_watt})", flush=True)
                else:
                    (pasued_gpu, res) = self.pause_one_gpu(is_be=True)
                    if res:
                        self.workloads_state[GPUWorkloadType.BE][pasued_gpu]['cap']['current'] = max_supported_power
                        self.resource_controller.set_freq(
                            app_name=self.be_workload_name,
                            gpu=pasued_gpu,
                            freq=225
                        )
                        amount_watt = amount_watt - capped_tos_power
                        if self.print_debug_info:
                            print(f"\t-[FRWorkloadController/decrease:{getframeinfo(currentframe()).lineno}]: no downroom but amount less than half of min, paused! (gpu:{pasued_gpu} amount:{amount_watt})", flush=True)
                    else:
                        if self.print_debug_info:
                            print(f"\t-[FRWorkloadController/decrease:{getframeinfo(currentframe()).lineno}]: no downroom but amount less than half of min, no gpu to pause! (amount_watt:{amount_watt})", flush=True)
                        break
        if got_in:
            self.save_plot_snapshot(target_power=target_power, current_power=current_power)

    def increase_gpu_power(self, target_power: int, current_power: int):
        amount_watt = target_power - current_power
        max_supported_power = int(self.power_model['be']['power2cap'][60]['max_supported'])
        min_supported_power = int(self.power_model['be']['power2cap'][60]['min_supported'])

        # Check if with current set of gpus we can reach target_power
        # curr_internal_powers = self.get_cur_internal_power()
        # if curr_internal_powers > target_power:
        #     if self.debug:
        #         print(f"\t-[FRWorkloadController/increase:{getframeinfo(currentframe()).lineno}]: cur internal power: {curr_internal_powers} target:{target_power}, skipping...", flush=True)
        #     self.save_plot_snapshot(target_power=target_power, current_power=current_power)
        #     return
        # else:
        #     amount_watt = target_power - curr_internal_powers
        # if self.debug:
        #     print(f"\t-[FRWorkloadController/increase:{getframeinfo(currentframe()).lineno}]: cur: {current_power} target:{target_power} internal power: {curr_internal_powers} requested amount to increase: {amount_watt}", flush=True)

        if amount_watt > 1 and len(self.be_capped_stack) == 0: # first time we have no gpus to work on
            if self.print_debug_info:
                print(f"\t-[FRWorkloadController/increase:{getframeinfo(currentframe()).lineno}]: stack empty, resuming a gpu", flush=True)
            (resumed, res) = self.resume_one_gpu(is_be=True)
            if res: # 
                self.workloads_state[GPUWorkloadType.BE][resumed]['cap']['current'] = min_supported_power
                self.resource_controller.set_freq(
                    app_name=self.be_workload_name,
                    gpu=resumed,
                    freq=int(min_supported_power)
                )
                amount_watt -= min_supported_power
                if self.print_debug_info:
                    print(f"\t-[FRWorkloadController/increase:{getframeinfo(currentframe()).lineno}]: resumed (gpu:{resumed} cap:{min_supported_power} amount:{amount_watt})", flush=True)
                if self.print_debug_info:
                    self.save_plot_snapshot(target_power=target_power, current_power=current_power)
            else:
                if self.print_debug_info:
                    print(f"\t-[FRWorkloadController/increase:{getframeinfo(currentframe()).lineno}]: No GPU found to start the stack!", flush=True)
                self.save_plot_snapshot(target_power=target_power, current_power=current_power)
                return
        got_in = False
        while amount_watt > 1 and len(self.be_capped_stack) > 0:
            got_in = True
            capped_tos = self.be_capped_stack[-1] # tos: top of stack
            # up room left to increase cap
            capped_tos_power = int(self.workloads_state[GPUWorkloadType.BE][capped_tos]['cap']['current'])
            up_room_left_on_tos = max_supported_power - capped_tos_power
            if self.print_debug_info:
                print(f"\t-[FRWorkloadController/increase:{getframeinfo(currentframe()).lineno}]: LOOP (tos:{capped_tos} cur_tos_power:{capped_tos_power} up_room:{up_room_left_on_tos} amount:{amount_watt})", flush=True)
            # 1
            if up_room_left_on_tos > 0: # increase whatever room left on tos first
                if amount_watt > up_room_left_on_tos: # not enough room to match cap with amount_watt => cap to max (rest is handled in next round of while loop)
                    self.workloads_state[GPUWorkloadType.BE][capped_tos]['cap']['current'] = max_supported_power
                    self.resource_controller.set_freq(
                        app_name=self.be_workload_name,
                        gpu=capped_tos,
                        freq=225
                    )
                    amount_watt -= up_room_left_on_tos
                    if self.print_debug_info:
                        print(f"\t-[FRWorkloadController/increase:{getframeinfo(currentframe()).lineno}]: no uproom left, tos to max! (gpu:{capped_tos} prev_cap:{capped_tos_power} new_cap:{max_supported_power} amount:{amount_watt})", flush=True)
                else: # there is enough room to match cap to amount_watt. Just cap to match amount
                    new_cap = capped_tos_power+amount_watt
                    self.workloads_state[GPUWorkloadType.BE][capped_tos]['cap']['current'] = new_cap
                    self.resource_controller.set_freq(
                        app_name=self.be_workload_name,
                        gpu=capped_tos,
                        freq=int(new_cap)
                    )
                    amount_watt = 0
                    if self.print_debug_info:
                        print(f"\t-[FRWorkloadController/increase:{getframeinfo(currentframe()).lineno}]: uproom cap! (gpu:{capped_tos} prev_cap:{capped_tos_power} new_cap:{new_cap} amount:{amount_watt})", flush=True)
                    # break
            else:  # no room left on current set of gpus to increase, add more gpus
                if amount_watt > max_supported_power: # increment is higher than only adding one gpu with max cap (add that, rest is going to be handled in the next round of the loop)
                    #TODO: add gpu with max cap value
                    (resumed_gpu, res) = self.resume_one_gpu(is_be=True)
                    if res:
                        self.workloads_state[GPUWorkloadType.BE][resumed_gpu]['cap']['current'] = max_supported_power
                        self.resource_controller.set_freq(
                            app_name=self.be_workload_name,
                            gpu=resumed_gpu,
                            freq=225
                        )
                        amount_watt -= max_supported_power
                        print(f"\t-[FRWorkloadController/increase]: new gpu with max cap needed, resume! (gpu:{resumed_gpu} cap={max_supported_power} amount:{amount_watt})", flush=True)
                    else:
                        if self.print_debug_info:
                            print(f"\t-[FRWorkloadController/increase:{getframeinfo(currentframe()).lineno}]: new gpu with max cap needed, failed! (amount:{amount_watt})", flush=True)
                        break
                    
                elif amount_watt > min_supported_power//2: # incerement is more than half-of-min_cap but less than max_cap, resume gpu 
                    (resumed_gpu, res) = self.resume_one_gpu(is_be=True)
                    if res: # cap newly resumed gpu to the higher value between (min_cap possible and the amount_requested)
                        new_cap = max(amount_watt, min_supported_power)
                        self.workloads_state[GPUWorkloadType.BE][resumed_gpu]['cap']['current'] = new_cap
                        self.resource_controller.set_freq(
                            app_name=self.be_workload_name,
                            gpu=resumed_gpu,
                            freq=int(new_cap)
                        )
                        amount_watt = 0
                        if self.print_debug_info:
                            print(f"\t-[FRWorkloadController/increase:{getframeinfo(currentframe()).lineno}]: new gpu with cap needed, resumed! (gpu:{resumed_gpu} cap:{new_cap} amount:{amount_watt})", flush=True)
                    else: # no gpu found to resume
                        if self.print_debug_info:
                            print(f"\t-[FRWorkloadController/increase:{getframeinfo(currentframe()).lineno}]: new gpu with cap needed, failed! (amount:{amount_watt})", flush=True)
                        break
                else: # # incerement is less than half of min_cap, do nothing since we can't do anything!
                    if self.print_debug_info:
                        print(f"\t-[FRWorkloadController/increase:{getframeinfo(currentframe()).lineno}]: amount is less than half of min cap, no operation! (amount:{amount_watt})", flush=True)
                    break

        if got_in:
            self.save_plot_snapshot(target_power=target_power, current_power=current_power)

    def get_unmasked_gpu(self): # to decrease and increase power
        """
        returns the gpu that has is not masked
        if two gpus are unmasked, returns lower ind one. 
        if no gpu is unmasked returns None
        """
        gpu_range = range(self.num_system_gpus)
        unmasked_gpu = None
        for gpu in gpu_range:
            if gpu not in list(self.workloads_state[GPUWorkloadType.BE].keys()): # not added
                continue
            if self.workloads_state[GPUWorkloadType.BE][gpu]['state'] != WorkloadState.RUNNING: # paused
                continue
            if self.workloads_state[GPUWorkloadType.BE][gpu]['cus'] < 60:
                continue
            unmasked_gpu = gpu
            break
        return unmasked_gpu

    def get_masked_gpu(self, state, return_lowest_masked): # to decrease and increase power
        """
        returns the gpu that has is masked with lowest number of cus
        if two gpus are masked, returns lower ind one. 
        if no gpu is unmasked returns None
        """
        gpu_range = range(self.num_system_gpus)
        mask = 61 if return_lowest_masked else 0
        masked_gpu = None
        for gpu in gpu_range:
            if gpu not in list(self.workloads_state[GPUWorkloadType.BE].keys()): # not added
                continue
            if self.workloads_state[GPUWorkloadType.BE][gpu]['state'] != state: # paused
                continue
            cus = self.workloads_state[GPUWorkloadType.BE][gpu]['cus']
            if return_lowest_masked:
                if cus < mask:
                    mask = cus
                    masked_gpu = gpu
            else:
                if cus > mask:
                    mask = cus
                    masked_gpu = gpu
        return (masked_gpu, mask)
    
    def get_uncapped_gpus(self): # to decrease and increase power
        """
        returns list of gpus that has is not capped. 
        If no uncapped found, returns empty list
        """
        gpu_range = range(self.num_system_gpus-1, -1, -1)
        uncapped_gpus = list()
        for gpu in gpu_range:
            if gpu not in list(self.workloads_state[GPUWorkloadType.BE].keys()): # not added
                continue
            if self.workloads_state[GPUWorkloadType.BE][gpu]['state'] != WorkloadState.RUNNING: # paused
                continue
            if self.workloads_state[GPUWorkloadType.BE][gpu]['cap'] < 225:
                continue
            uncapped_gpus.append(gpu)

        return uncapped_gpus

    def get_capped_gpu(self, return_highest_cap): # to decrease power
        """
        returns the gpu that has lowest capped value and cap as tuple (cap_val, gpu)
        if two gpus with equal cap, returns lower ind one. 
        if non gpu is capped returns (None, 225)
        """
        gpu_range = range(self.num_system_gpus)
        cap = 0 if return_highest_cap else 226
        capped_gpu = None
        for gpu in gpu_range:
            if gpu not in list(self.workloads_state[GPUWorkloadType.BE].keys()): # not added
                continue
            if self.workloads_state[GPUWorkloadType.BE][gpu]['state'] != WorkloadState.RUNNING: # paused
                continue
            cur_gpu_cap = self.workloads_state[GPUWorkloadType.BE][gpu]['cap']
            if return_highest_cap:
                if cur_gpu_cap > cap:
                    cap = cur_gpu_cap
                    capped_gpu = gpu
            else:
                if cur_gpu_cap < cap:
                    cap = cur_gpu_cap
                    capped_gpu = gpu

        return (capped_gpu, cap)

    def add_one_gpu(self, is_be):
        success = False
        added_gpu = None
        args = dict()
        workload_type = GPUWorkloadType.BE if is_be else GPUWorkloadType.LC
        worload_name = self.be_workload_name if is_be else self.lc_workload_name
        # Check if there is any GPU left 
        if len(list(self.workloads_state[workload_type].keys())) == self.num_system_gpus:
            return (added_gpu, success)
        
        # [7,6,5,4,3,2,1,0] for BE. [0,1,2,3,4,5,6,7] for LC
        gpu_range = range(self.num_system_gpus-1, -1,-1) if is_be else range(self.num_system_gpus)
        for gpu in gpu_range: # reverse iteratation on all gpus 
            if gpu in list(self.workloads_state[workload_type].keys()): # already added
                continue
            args['gpu'] = str(gpu)
            args['model'] = str(self.lc_model_name) # only used for lc
            args['batch_size'] = str(self.lc_batch_size) # only used for lc
            self.workload_runner.add_gpu(
                is_be_wl=is_be, 
                args=args
            )
            if is_be is False:
                assert self.resource_controller.add_cu(
                    app_name=worload_name,
                    gpu=gpu,
                    cus=60,
                    is_be=is_be
                ), f"Could not allocate all cus on gpu {gpu} for be={is_be}"
            elif gpu not in self.workloads_state[GPUWorkloadType.LC]:
                    assert self.resource_controller.add_cu(
                        app_name=worload_name,
                        gpu=gpu,
                        cus=60,
                        is_be=is_be
                    ), f"Could not allocate all cus on gpu {gpu} for be={is_be}"
            self.workloads_state[workload_type][gpu] = {
                'state': WorkloadState.RUNNING,
                'cap': {
                    'current': int(self.power_model['be']['power2cap'][60]['max_supported']),
                    'max': int(self.power_model['be']['power2cap'][60]['max_supported']),
                    'min':  int(self.power_model['be']['power2cap'][60]['min_supported'])
                },
                'cus': 60
            }
            print(f"\t-[FRWorkloadController/add_one_gpu]: gpu {gpu} added to {workload_type} with status {WorkloadState.RUNNING}", flush=True)
            added_gpu = gpu
            success = True
            break # return after finding the first non-BE gpu
        return (added_gpu, success)

    def pause_one_gpu(self, is_be):
        success = False
        paused_gpu = None
        args = dict()
        workload_type = GPUWorkloadType.BE if is_be else GPUWorkloadType.LC
        # Check if worklaod is loaded on any GPU 
        if len(list(self.workloads_state[workload_type].keys())) == 0:
            return (paused_gpu, success)

        # [7,6,5,4,3,2,1,0] for LC. [0,1,2,3,4,5,6,7] for BE
        gpu_range = range(self.num_system_gpus) if is_be else range(self.num_system_gpus-1, -1,-1)
        for gpu in gpu_range: # reverse iteratation on all gpus 
            if gpu not in list(self.workloads_state[workload_type].keys()): # does not exist
                continue
            if self.workloads_state[workload_type][gpu]['state'] == WorkloadState.PAUSED: # paused already
                continue
            args['gpu'] = str(gpu)
            args['model'] = str(self.lc_model_name) # only used for lc
            self.workload_runner.pause_gpu(
                is_be_wl=is_be, 
                args=args
            )
            self.workloads_state[workload_type][gpu]['state'] = WorkloadState.PAUSED

            if gpu in self.be_capped_stack:
                self.be_capped_stack.remove(gpu)
            
            print(f"\t-[FRWorkloadController/pause_one_gpu]: gpu {gpu} for {workload_type} was paused", flush=True)
            success = True
            paused_gpu = gpu
            break # return after finding the first non-BE gpu 
        return (paused_gpu, success)

    def resume_one_gpu(self, is_be):
        success = False
        resumed_gpu = None
        args = dict()
        workload_type = GPUWorkloadType.BE if is_be else GPUWorkloadType.LC
        # Check if worklaod is loaded on any GPU 
        if len(list(self.workloads_state[workload_type].keys())) == 0:
            return (resumed_gpu, success)

        # [7,6,5,4,3,2,1,0] for BE. [0,1,2,3,4,5,6,7] for LC
        gpu_range = range(self.num_system_gpus-1, -1,-1) if is_be else range(self.num_system_gpus)
        for gpu in gpu_range: # reverse iteratation on all gpus 
            if gpu not in list(self.workloads_state[workload_type].keys()): # does not exist
                continue
            if self.workloads_state[workload_type][gpu]['state'] == WorkloadState.RUNNING: # already running
                continue
            args['gpu'] = str(gpu)
            args['model'] = str(self.lc_model_name) # only used for lc
            self.workload_runner.resume_gpu(
                is_be_wl=is_be, 
                args=args
            )
            self.workloads_state[workload_type][gpu]['state'] = WorkloadState.RUNNING
            self.be_capped_stack.append(gpu)
            print(f"\t-[FRWorkloadController/resume_one_gpu]: gpu {gpu} for {workload_type} was resumed", flush=True)
            success = True
            resumed_gpu = gpu
            break # return after finding the first non-BE gpu 
        return (resumed_gpu, success)

    def save_be_stats(self, args):
        assert 'stat_file' in args, f"[FRWorkloadController/cleanup_and_save_be_stats]: args {args} must have a valid stat_file"
        self.workload_runner.finsh_wl(args)
    
    def start_services(self):
        if self.docker_runner.start_docker(self.lc_workload_name) is False:
            if self.print_debug_info:
                print(f"\t-[FRWorkloadController/start_services]: Failed starting lc workload: {self.lc_workload_name}!")
            return False
        if self.docker_runner.start_docker(self.be_workload_name) is False:
            if self.print_debug_info:
                print(f"\t-[FRWorkloadController/start_services]: Failed starting be workload: {self.be_workload_name}!")
            return False
        if self.docker_runner.start_docker('power-broadcaster') is False:
            if self.print_debug_info:
                print("\t-[FRWorkloadController/start_services]: Failed starting power-broadcaster!")
            return False
        return self.reset_gpu_caps()

    def stop_services(self):
        self.reset_gpu_caps()
        if self.docker_runner.stop_docker(self.lc_workload_name) is False:
            if self.print_debug_info:
                print(f"\t-[FRWorkloadController/stop_services]: Failed stopping lc workload: {self.lc_workload_name}!")
            return False
        if self.docker_runner.stop_docker(self.be_workload_name) is False:
            if self.print_debug_info:
                print(f"\t-[FRWorkloadController/stop_services]: Failed stopping be workload: {self.be_workload_name}!")
            return False
        if self.docker_runner.stop_docker('power-broadcaster') is False:
            if self.print_debug_info:
                print("\t-[FRWorkloadController/stop_services]: Failed stopping power-broadcaster!")
            return False
        return True
    def reset_gpu_caps(self):
        for g in range(self.num_system_gpus):
            self.resource_controller.set_freq(app_name=self.be_workload_name, gpu=g, freq=225)
        
        if self.print_debug_info:
            print(f"\t-[FRWorkloadController/reset_gpu_caps]: gpu caps reset done!")
        return True

    def reset_gpu_cus(self):
        for wl in [GPUWorkloadType.BE, GPUWorkloadType.LC]:
            is_be = True if wl == GPUWorkloadType.BE else False
            app_name = self.be_workload_name if is_be else self.lc_workload_name
            for g in self.workloads_state[wl]:
                cur_cus = self.resource_controller.get_current_cus(gpu=g, is_be=is_be)
                self.resource_controller.remove_cu(app_name=app_name, gpu=g, cus =cur_cus, is_be=is_be)
            if self.print_debug_info:
                print(f"\t-[FRWorkloadController/reset_gpu_cus]: gpu cus reset for workload {app_name} done!")

    def initialize_workloads(self):
        # starting the servers (no gpu will be assigned, no running)
        self.workload_runner.start(is_be_wl=False)
        self.workload_runner.start(is_be_wl=True)
        if self.print_debug_info:
            print(f"\t-[FRWorkloadController/initialize_workloads]: Workloads intialized!")
        return True

    def cleanup_remote_and_save_results(self, save_path: str, chunk: int):
        # IMPORTANT: order is important dont move them around
        if self.print_debug_info:
            print(f"\t-[FRWorkloadController/cleanup_remote_and_save_results]: snapshot of states for chunk {chunk} save in {save_path}/states_{chunk}.pickle!")
        if chunk >= 0:
            with open(f"{save_path}/states_{chunk}.pickle", 'wb') as handle:
                pickle.dump(self.plot_data, handle)
        if self.print_debug_info:
            print(f"\t-[FRWorkloadController/pause_one_gpu]: Stopping services!")
        assert self.stop_services(), "Could not cleanup the remote"
        if self.print_debug_info:
            print(f"\t-[FRWorkloadController/pause_one_gpu]: resetting gpu cus!")
        self.reset_gpu_cus()
        for workload in [GPUWorkloadType.BE, GPUWorkloadType.LC]:
            self.workloads_state[workload] = dict()
        self.be_capped_stack = list()
        self.plot_data = list()
        if self.print_debug_info:
            print(f"\t-[FRWorkloadController/cleanup_remote_and_save_results]: cleanup and save done!")

    def setup_remote(self):
        # assert self.lc_avg_load_pct > 0 and self.lc_avg_load_pct <= 100, f"LC load must be in range [0,100] given {self.lc_avg_load_pct}"
        if self.print_debug_info:
            print(f"\t-[FRWorkloadController/setup_remote]: setting up remote services!")
        # Dockers
        assert self.start_services(), f"Failed to start services on the traget"
        # Workloads
        assert self.initialize_workloads(), f"Failed to initialize workloads on the traget"
        
        # Convert avg load % to number of needed GPUs
        self.lc_num_max_gpus_needed = math.ceil(max(self.lc_load_pct_list)/100.0 * self.num_system_gpus)

        # Loading workloads into GPUs for each workload
        # 1. LC: on all but pause unnecessary ones
        if self.print_debug_info:
            print(f"\t-[FRWorkloadController/setup_remote]: adding workloads to the remote server with max load: {max(self.lc_load_pct_list)}% gpus needed for lc: {self.lc_num_max_gpus_needed}", flush=True)
        for g in range(self.lc_num_max_gpus_needed):
            assert self.add_one_gpu(is_be=False), f"Failed adding ({g}/{self.num_system_gpus}) gpu to LC"
            # if g >= self.lc_num_gpus_needed:
            #     assert self.pause_one_gpu(is_be=False), f"Failed pausing ({g}/{self.num_system_gpus}) gpu to LC"
        # BE: on all and all paused
        for g in range(self.num_system_gpus):
            assert self.add_one_gpu(is_be=True), f"Failed adding ({g}/{self.num_system_gpus}) gpu to BE"
            assert self.pause_one_gpu(is_be=True), f"Failed pausing ({g}/{self.num_system_gpus}) gpu to BE"
        if self.print_debug_info:
            print(f"\t-[FRWorkloadController/setup_remote]: warming up LC({self.lc_workload_name})")
        self.workload_runner.run_lc_client(
            warmp_first=True,
            num_warmpup_load_steps=3,
            warmup_step_duration_sec=20,
            gpus=self.lc_num_gpus_needed,
            max_rps_per_gpu=75,
            trace_file="",
            trace_unit_sec=60,
            no_run=True
        )
        if self.print_debug_info:
            print(f"\t-[FRWorkloadController/setup_remote]: BE({self.be_workload_name}-0:{self.num_system_gpus-1}) and LC({self.lc_workload_name}-0:{self.lc_num_gpus_needed-1}) are ready!")
        return True

    def adjust_resources_cap_only(self, next_power: float, current_power: float):
        # 1- map next_power to resource with power model using a policy 
        #  - increase power (in order)
        #       1- inc LC resources (gpu/cap/cu)? 
        #       2- in BE resources (gpu/cap/cu)?
        #  - decrease power (in order):
        #       1- decrease BE resrouces (gpu, cap, cu)?
        #       2- decrease LC resources to safe point (gpu, cap, cu)?
        # Notes: check resrouce overlap
        if self.print_debug_info:
            print(f"\t-[FRWorkloadController/adjust_resources_cap_only]: cur={current_power} next={next_power}", flush=True)
        # cur_internal_power = self.get_cur_internal_power()
        # increase_power = cur_internal_power < next_power
        start_time = time.time()
        increase_power = current_power < next_power
        if increase_power:
            self.increase_gpu_power(current_power=current_power, target_power=next_power)
        else:
            self.decrease_gpu_power(current_power=current_power, target_power=next_power)
        end_time = time.time()
        elapsed = end_time-start_time
        if self.print_debug_info: 
            print(f"\t-[FRWorkloadController/adjust_resources_cap_only]--------------------------------------------------{elapsed}", flush=True)
        
        return elapsed

    def plot_states(self, post_fix: str = "stack", save_path: str= f"{os.path.dirname(os.path.abspath(__file__))}/../data/"):
        import matplotlib.pyplot as plt
        import numpy as np
        if self.print_debug_info:
            print(f"\t-[FRWorkloadController/plot_stack]: plotting states!")
        targets = list()
        currents = list()
        gpus_power = dict()
        x_ax_data = list()
        lc_load_data = list()
        lc_x_data = list()
        
        ratio = math.floor(len(self.plot_data) / len(self.lc_load_pct_list))
        ind = 0
        for lc_load in self.lc_load_pct_list:
            for i in range(ratio):
                lc_load_data.append((lc_load/100.0)*200)
                lc_x_data.append(ind)
                ind += 1
        for ind in range(len(self.plot_data)):
            x_ax_data.append(ind)
            data = self.plot_data[ind]
            targets.append(data['target'])
            currents.append(data['current'])
            for g in self.workloads_state[GPUWorkloadType.BE]:
                if g not in gpus_power:
                    gpus_power[g] = list()
                if g not in data['capp_stack']:
                    gpus_power[g].append(0.0)
                else:    
                    gpu_power = data['wl_state'][GPUWorkloadType.BE][g]['cap']['current']
                    assert data['wl_state'][GPUWorkloadType.BE][g]['state'] == WorkloadState.RUNNING, f"GPU {g} found in cap_stack but not runnig stack:{data['wl_state'][GPUWorkloadType.BE][g]}"
                    gpus_power[g].append(float(gpu_power))

        for i, g in enumerate(self.workloads_state[GPUWorkloadType.BE]):
            gpu_np = np.array(gpus_power[g], dtype='f')
            accum = np.zeros_like(gpu_np, dtype='f')
            # add prev. gpus power to this as offset 
            for j in range(min(self.workloads_state[GPUWorkloadType.BE]), max(self.workloads_state[GPUWorkloadType.BE])+1):
                if j > g:
                    accum += np.array(gpus_power[j], dtype='f')
            plt.bar(np.array(x_ax_data)+0.5, gpu_np, width=1, bottom=accum, label=f"gpu{g}", ec='k')
        plt.step(np.array(x_ax_data)+0.5, targets, where='mid', label='Target', c='red')
        plt.step(np.array(x_ax_data)+0.5, currents, where='mid', label='Current', c='blue')
        plt.step(np.array(lc_x_data)+0.5, lc_load_data, where='mid', label='LC Load', c='k')
        plt.legend(loc='upper center', ncol=5)
        # Save in png and pdf in provided log_dir
        plt.xlabel("Time")
        plt.ylabel("Power (watt)")
        plt.ylim([0,max(targets)+400])
        if self.print_debug_info:
            print(f"\t-[FRWorkloadController/plot_stack]: saved states plot to: {save_path}/{post_fix}.png")
        plt.savefig(f"{save_path}/{post_fix}.png")
        plt.savefig(f"{save_path}/{post_fix}.pdf")
        plt.close()

    def run_lc(self, unit_sec: int = 60):
        # if self.debug:
        #     return
        self.lc_client_runner = LCClientRunner(
            trace_file=self.lc_trace_file,
            gpus=self.num_system_gpus,
            max_rps_per_gpu=75,
            trace_unit_sec=unit_sec,
            debug=False
        )
        self.lc_client_runner.run_client(
            cmd="",
            server_ip=self.target_ip,
            blocking=False)
        if self.print_debug_info:
            print(f"\t-[FRWorkloadController/run_lc]: running lc client (units per pct:{unit_sec})")
    
    def get_lc_results(self):
        if self.print_debug_info:
            print(f"\t-[FRWorkloadController/get_lc_results]: collecting lc results")
        return self.lc_client_runner.get_lc_results()
    
    def __del__(self):
        # self.cleanup_remote_and_save_results(save_path="/tmp", chunk=-1)
        del self.resource_controller
        del self.docker_runner
