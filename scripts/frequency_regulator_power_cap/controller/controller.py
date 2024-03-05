import os
import sys
import math
import enum
import copy
import datetime
import concurrent.futures
sys.path.insert(1, '../../') # for utils TODO: make it a wheel

from utils.stat_collectors.lc_load_collector import LCLoadCollector
from scripts.utils.resource_managers.resource_manager_distributed import GPUResourceManagerDistributed, GPUWorkloadType
from utils.workload_runners.workload_runner_remote import RemoteDockerRunner, RemoteWorkloadRunner

class WorkloadState(enum.Enum):
    PAUSED = 0
    RUNNING = 1

class FRSingleGPUWorkloadController:
    num_system_gpus = 8
    be_stat_base_dir = 'be_stats/'
    lc_thread = None
    lc_num_gpus_needed = 1
    lc_qos = None
    workloads_state = dict()
    free_gpus = list()
    def __init__(
            self,
            power_model,
            lc_model_name: str = "resnet152",
            lc_batch_size: int = 1,
            lc_workload_name: str = "Inference-Server",
            be_workload_name: str= "miniMDock",
            target_ip: str = "172.20.0.9",
            remote_ip: str = "172.20.0.6",
            remote_resource_ctl_port: str = "5000",
            remote_workload_ctl_port: str = "3000",
            target_docker_ctrl_port: str = "4000",
            debug = False
        ):
        self.debug = debug
        self.RESULTS_DIR = "/workspace/fr_results"
        self.LOGS_DIR = "/workspace/experiment_logs"
        self.target_ip = target_ip
        self.remote_ip = remote_ip
        self.power_model = power_model
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
            debug=debug
        )

        self.workload_runner = RemoteWorkloadRunner(
            remote_ip=self.remote_ip,
            target_ip=self.target_ip,
            remote_workload_control_port=self.remote_workload_ctl_port,
            lc_workload_name=lc_workload_name,
            be_workload_name=be_workload_name,
            wait_after_send=False,
            debug=debug
        )
        
        self.resource_controller = GPUResourceManagerDistributed(
            remote_control_ip=remote_ip,
            remote_control_port=remote_resource_ctl_port,
            debug=debug
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
        # get_uncapped:
        #   yes: cap it
        #   no: get_capped(return_highest_cap=True):
        #       yes: cap more
        #       no: get_masked(state=running, return_lowest_masked=false):
        #           yes: mask more
        #           no: get_masked(state=running, return_lowest_masked=True)
        #               yes: pause
        #               no: maybe nothing else can be done?!
        # min_supported_power = be_pm['power2cap'][60]['min_supported']
        # max_supported_power = be_pm['power2cap'][60]['max_supported']
        # max_decrease_range = max_supported_power - min_supported_power
        # uncapped_gpus = self.get_uncapped_gpus()
        # while True:
        #     if max_decrease_range * len(uncapped_gpus) < amount_watt:
        #         # All gpus need to capped to min_supported_power
        #         cap_to = int(be_pm['power2cap'][60]['reg_model']([[min_supported_power]]))
        #         cap_to = min(225, cap_to)
        #         cap_to = max(10, cap_to)
        #         # Set cap
        #         self.resource_controller.set_freq(
        #             app_name=self.be_workload_name,
        #             gpu=uncapped_gpus[0],
        #             freq=cap_to
        #         )
        #         # Update state
        #         self.workloads_state[GPUWorkloadType.BE][gpu]['cap'] = cap_to
        #         # Update power
        #         amount_watt -= len(uncapped_gpus) * (max_supported_power - min_supported_power)
        #         break
        #     else:
        #         pass
        amount_watt =  current_power - target_power
        max_supported_power = int(self.power_model['be']['power2cap'][60]['max_supported'])
        min_supported_power = int(self.power_model['be']['power2cap'][60]['min_supported'])

        # Check if with current set of gpus we can reach target_power
        curr_internal_powers = self.get_cur_internal_power()
        if curr_internal_powers <= target_power:
            print(f"[FRWorkloadController/decrease]: cur internal power: {curr_internal_powers} target:{target_power}, skipping...", flush=True)
            if self.debug:
                    self.save_plot_snapshot(target_power=target_power, current_power=current_power)
            return
        else:
            amount_watt =  curr_internal_powers - target_power
        print(f"[FRWorkloadController/decrease]: cur: {current_power} target:{target_power} internal power: {curr_internal_powers} requested amount to increase: {amount_watt}", flush=True)
        
        # 1- increase cap on existing GPUs
        # Notes:
        #   Only top of stack,i.e. tos, is capped! 
        #   Others are not capped
        # 2- Then add gpu if needed
        got_in = False
        while amount_watt > 1 and len(self.be_capped_stack) > 0:
            got_in = True
            capped_tos = self.be_capped_stack[-1] # tos: top of stack
            # up room left to increase cap
            capped_tos_power = int(self.workloads_state[GPUWorkloadType.BE][capped_tos]['cap']['current'])
            down_room_left_on_tos = capped_tos_power - min_supported_power
            print(f"[FRWorkloadController/decrease]: LOOP tos:{capped_tos} cur_tos_power:{capped_tos_power} down_room:{down_room_left_on_tos} AMOUNT={amount_watt}", flush=True)
            # 1
            if amount_watt > capped_tos_power:
                (pasued_gpu, res) = self.pause_one_gpu(is_be=True)
                if res:
                    self.workloads_state[GPUWorkloadType.BE][pasued_gpu]['cap']['current'] = max_supported_power
                    self.resource_controller.set_freq(
                        app_name=self.be_workload_name,
                        gpu=pasued_gpu,
                        freq=max_supported_power
                    )
                    amount_watt -= capped_tos_power
                    print(f"[FRWorkloadController/decrease]: check#2.1 gpu {pasued_gpu} paused with cap={capped_tos_power} AMOUNT={amount_watt}", flush=True)
                else:
                    print("\t-[FRWorkloadController/decrease]: check#2 No GPU found to pasue, amount_watt to increase stays the same", flush=True)
                    break
            else:
                if down_room_left_on_tos > 0: #down_room_left_on_tos
                    # TODO: capp capped_tos to to max supported capp
                    if amount_watt > down_room_left_on_tos:
                        new_cap = min_supported_power
                        self.workloads_state[GPUWorkloadType.BE][capped_tos]['cap']['current'] = new_cap
                        self.resource_controller.set_freq(
                            app_name=self.be_workload_name,
                            gpu=capped_tos,
                            freq=new_cap
                        )
                        amount_watt -= down_room_left_on_tos
                        print(f"[FRWorkloadController/decrease]: check#1.1 gpu {capped_tos} cap changed from {capped_tos_power} to {min_supported_power} AMOUNT={amount_watt}", flush=True)
                    else:
                        new_cap = capped_tos_power - amount_watt
                        self.workloads_state[GPUWorkloadType.BE][capped_tos]['cap']['current'] = new_cap
                        self.resource_controller.set_freq(
                            app_name=self.be_workload_name,
                            gpu=capped_tos,
                            freq=new_cap
                        )
                        amount_watt = 0
                        print(f"[FRWorkloadController/decrease]: check#1.1 gpu {capped_tos} cap changed from {capped_tos_power} to {min_supported_power} AMOUNT={amount_watt}", flush=True)
                elif amount_watt < capped_tos_power - min_supported_power//2:
                    # TODO: capp capped_tos to amount_watt
                    new_cap = min_supported_power
                    self.workloads_state[GPUWorkloadType.BE][capped_tos]['cap']['current'] = new_cap
                    self.resource_controller.set_freq(
                        app_name=self.be_workload_name,
                        gpu=capped_tos,
                        freq=new_cap
                    )
                    amount_watt = 0
                    print(f"[FRWorkloadController/decrease]: check#1.2 gpu {capped_tos} cap changed from {capped_tos_power} to {new_cap} AMOUNT={amount_watt}", flush=True)
                    # break
                else:
                    (pasued_gpu, res) = self.pause_one_gpu(is_be=True)
                    if res:
                        self.workloads_state[GPUWorkloadType.BE][pasued_gpu]['cap']['current'] = max_supported_power
                        self.resource_controller.set_freq(
                            app_name=self.be_workload_name,
                            gpu=pasued_gpu,
                            freq=max_supported_power
                        )
                        amount_watt = 0 
                        print(f"[FRWorkloadController/decrease]: check#2.1 gpu {pasued_gpu} paused with cap={capped_tos_power} AMOUNT={amount_watt}", flush=True)
                    else:
                        print("\t-[FRWorkloadController/decrease]: check#2 No GPU found to pasue, amount_watt to increase stays the same", flush=True)
                        break
        if self.debug and got_in:
                    self.save_plot_snapshot(target_power=target_power, current_power=current_power)

    def increase_gpu_power(self, target_power: int, current_power: int):
        # get_masked(state=paused, return_lowest_masked=True):
        #   yes: resume
        #   no: get_masked(state=running,return_lowest_masked=True):
        #       yes: increase cu
        #       no: get_capped(return_highest_cap=False)
        #           yes: uncap
        #           no: maybe nothing else can be done?!
        amount_watt = target_power - current_power
        max_supported_power = int(self.power_model['be']['power2cap'][60]['max_supported'])
        min_supported_power = int(self.power_model['be']['power2cap'][60]['min_supported'])

        # Check if with current set of gpus we can reach target_power
        curr_internal_powers = self.get_cur_internal_power()
        if curr_internal_powers > target_power:
            print(f"[FRWorkloadController/increase]: cur internal power: {curr_internal_powers} target:{target_power}, skipping...", flush=True)
            if self.debug:
                    self.save_plot_snapshot(target_power=target_power, current_power=current_power)
            return
        else:
            amount_watt = target_power - curr_internal_powers
        print(f"[FRWorkloadController/increase]: cur: {current_power} target:{target_power} internal power: {curr_internal_powers} requested amount to increase: {amount_watt}", flush=True)
        
        # 1- increase cap on existing GPUs
        # Notes:
        #   Only top of stack,i.e. tos, is capped! 
        #   Others are not capped
        # 2- Then add gpu if needed
        if amount_watt > 1 and len(self.be_capped_stack) == 0:
            print(f"[FRWorkloadController/increase]: stack empty, resuming a gpu", flush=True)
            (resumed, res) = self.resume_one_gpu(is_be=True)
            if res:
                self.workloads_state[GPUWorkloadType.BE][resumed]['cap']['current'] = min_supported_power
                self.resource_controller.set_freq(
                    app_name=self.be_workload_name,
                    gpu=resumed,
                    freq=min_supported_power
                )
                amount_watt -= min_supported_power
                print(f"[FRWorkloadController/increase]: gpu {resumed} was resumed, AMOUNT={amount_watt}", flush=True)
                if self.debug:
                    self.save_plot_snapshot(target_power=target_power, current_power=current_power)
            else:
                print("\t-[FRWorkloadController/increase]: No GPU found to resume, capped gpu stack is still empty", flush=True)
                if self.debug:
                    self.save_plot_snapshot(target_power=target_power, current_power=current_power)
                return
        got_in = False
        while amount_watt > 1 and len(self.be_capped_stack) > 0:
            got_in = True
            capped_tos = self.be_capped_stack[-1] # tos: top of stack
            # up room left to increase cap
            capped_tos_power = int(self.workloads_state[GPUWorkloadType.BE][capped_tos]['cap']['current'])
            up_room_left_on_tos = max_supported_power - capped_tos_power
            print(f"[FRWorkloadController/increase]: LOOP tos:{capped_tos} cur_tos_power:{capped_tos_power} up_room:{up_room_left_on_tos} AMOUNT={amount_watt}", flush=True)
            # 1
            if up_room_left_on_tos > 0:
                if amount_watt > up_room_left_on_tos:
                    # TODO: capp capped_tos to to max supported capp
                    self.workloads_state[GPUWorkloadType.BE][capped_tos]['cap']['current'] = max_supported_power
                    self.resource_controller.set_freq(
                        app_name=self.be_workload_name,
                        gpu=capped_tos,
                        freq=max_supported_power
                    )
                    amount_watt -= up_room_left_on_tos
                    print(f"[FRWorkloadController/increase]: check#1.1 gpu {capped_tos} cap changed from {capped_tos_power} to {max_supported_power} AMOUNT={amount_watt}", flush=True)
                else:
                    # TODO: capp capped_tos to amount_watt
                    new_cap = capped_tos_power+amount_watt
                    self.workloads_state[GPUWorkloadType.BE][capped_tos]['cap']['current'] = new_cap
                    self.resource_controller.set_freq(
                        app_name=self.be_workload_name,
                        gpu=capped_tos,
                        freq=new_cap
                    )
                    amount_watt = 0
                    print(f"[FRWorkloadController/increase]: check#1.2 gpu {capped_tos} cap changed from {capped_tos_power} to {new_cap} AMOUNT={amount_watt}", flush=True)
                    # break
            else:
                if amount_watt > max_supported_power:
                    #TODO: add gpu with max cap value
                    (resumed_gpu, res) = self.resume_one_gpu(is_be=True)
                    if res:
                        self.workloads_state[GPUWorkloadType.BE][resumed_gpu]['cap']['current'] = max_supported_power
                        self.resource_controller.set_freq(
                            app_name=self.be_workload_name,
                            gpu=resumed_gpu,
                            freq=max_supported_power
                        )
                    else:
                        print("\t-[FRWorkloadController/increase]: check#2 No GPU found to resume, amount_watt to increase stays the same", flush=True)
                        break
                    amount_watt -= max_supported_power
                    print(f"[FRWorkloadController/increase]: check#2.1 gpu {resumed_gpu} resumed with cap={max_supported_power} AMOUNT={amount_watt}", flush=True)
                elif amount_watt > min_supported_power//2:
                    #TODO: add gpu with cap = amount_watt
                    new_cap = max(amount_watt, min_supported_power)
                    (resumed_gpu, res) = self.resume_one_gpu(is_be=True)
                    if res:
                        self.workloads_state[GPUWorkloadType.BE][resumed_gpu]['cap']['current'] = new_cap
                        self.resource_controller.set_freq(
                            app_name=self.be_workload_name,
                            gpu=resumed_gpu,
                            freq=new_cap
                        )
                        amount_watt = 0
                        print(f"[FRWorkloadController/increase]: check#2.2 gpu {resumed_gpu} resumed with cap={new_cap} AMOUNT={amount_watt}", flush=True)
                    else:
                        print("\t-[FRWorkloadController/increase]: No GPU found to resume, amount_watt to increase stays the same", flush=True)
                        break
                else:
                    break
                    # new_cap = min_supported_power
                    # self.workloads_state[GPUWorkloadType.BE][capped_tos]['cap']['current'] = new_cap
                    # self.resource_controller.set_freq(
                    #     app_name=self.be_workload_name,
                    #     gpu=capped_tos,
                    #     freq=new_cap
                    # )
                    # amount_watt = 0
                    print(f"[FRWorkloadController/increase]: check#2.3 gpu {capped_tos} capped to cap={new_cap} AMOUNT={amount_watt}", flush=True)

        if self.debug and got_in:
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
                    app_name=self.be_workload_name,
                    gpu=gpu,
                    cus=60,
                    is_be=is_be
                ), f"Could not allocate all cus on gpu {gpu} for be={is_be}"
            elif gpu not in self.workloads_state[GPUWorkloadType.LC]:
                    assert self.resource_controller.add_cu(
                        app_name=self.be_workload_name,
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
            print(f"[FRWorkloadController/add_one_gpu]: gpu {gpu} added to {workload_type} with status {WorkloadState.RUNNING}", flush=True)
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
            
            print(f"[FRWorkloadController/pause_one_gpu]: gpu {gpu} for {workload_type} was paused", flush=True)
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
            print(f"[FRWorkloadController/resume_one_gpu]: gpu {gpu} for {workload_type} was resumed", flush=True)
            success = True
            resumed_gpu = gpu
            break # return after finding the first non-BE gpu 
        return (resumed_gpu, success)

    def cleanup_and_save_be_stats(self, args):
        assert 'stat_file' in args, f"[FRWorkloadController/cleanup_and_save_be_stats]: args {args} must have a valid stat_file"
        self.workload_runner.finsh_wl(args)
    
    def start_services(self):
        if self.docker_runner.start_docker(self.be_workload_name) is False:
            print(f"Failed starting be workload: {self.be_workload_name}!")
            return False
        if self.docker_runner.start_docker(self.lc_workload_name) is False:
            print(f"Failed starting lc workload: {self.lc_workload_name}!")
            return False
        if self.docker_runner.start_docker('power-broadcaster') is False:
            print("Failed starting power-broadcaster!")
            return False
        return True

    def initialize_workloads(self):
        # starting the servers (no gpu will be assigned, no running)
        self.workload_runner.start(is_be_wl=True)
        self.workload_runner.start(is_be_wl=False)
        return True

    def setup_remote(self, lc_avg_load_pct: int):
        assert lc_avg_load_pct > 0 and lc_avg_load_pct <= 100, f"LC load must be in range [0,100] given {lc_avg_load_pct}"
        
        # Dockers
        assert self.start_services(), f"Failed to start services on the traget"
        # Workloads
        assert self.initialize_workloads(), f"Failed to initialize workloads on the traget"
        
        # Convert avg load % to number of needed GPUs
        self.lc_num_gpus_needed = math.ceil(lc_avg_load_pct/100.0 * self.num_system_gpus)

        # Loading workloads into GPUs for each workload
        # 1. LC: on all but pause unnecessary ones
        print(f"Adding workloads to the remote server with avg load: {lc_avg_load_pct}% gpus needed for lc: {self.lc_num_gpus_needed}", flush=True)
        for g in range(self.num_system_gpus):
            assert self.add_one_gpu(is_be=False), f"Failed adding ({g}/{self.num_system_gpus}) gpu to LC"
            if g >= self.lc_num_gpus_needed:
                assert self.pause_one_gpu(is_be=False), f"Failed pausing ({g}/{self.num_system_gpus}) gpu to LC"
        # BE: on all and all paused
        for g in range(self.num_system_gpus):
            assert self.add_one_gpu(is_be=True), f"Failed adding ({g}/{self.num_system_gpus}) gpu to BE"
            assert self.pause_one_gpu(is_be=True), f"Failed pausing ({g}/{self.num_system_gpus}) gpu to BE"

        print(f"Warming up LC({self.lc_workload_name})")
        self.workload_runner.run_lc_client(
            warmp_first=True,
            num_warmpup_load_steps=5,
            warmup_step_duration_sec=60,
            gpus=self.lc_num_gpus_needed,
            max_rps_per_gpu=98,
            trace_file="",
            trace_unit_sec=60,
            no_run=True
        )
        print(f"BE({self.be_workload_name}-0:{self.num_system_gpus-1}) and LC({self.lc_workload_name}-0:{self.lc_num_gpus_needed-1}) are ready!")
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
        print(f"adjust_resources_cap_only: cur={current_power} next={next_power}")
        cur_internal_power = self.get_cur_internal_power()
        increase_power = cur_internal_power < next_power
        if increase_power:
            self.increase_gpu_power(current_power=current_power, target_power=next_power)
        else:
            self.decrease_gpu_power(current_power=current_power, target_power=next_power)

        # if self.debug:
        #     self.plot_stack()
        
    def plot_stack(self, save_path: str= f"{os.path.dirname(os.path.abspath(__file__))}/../data/"):
        import matplotlib.pyplot as plt
        import numpy as np
        
        targets = list()
        currents = list()
        gpus_power = dict()
        x_ax_data = list()
        print(len(self.plot_data))
        for ind in range(len(self.plot_data)):
            x_ax_data.append(ind)
            data = self.plot_data[ind]
            targets.append(data['target'])
            currents.append(data['current'])
            for g in self.workloads_state[GPUWorkloadType.BE]:
                if g not in gpus_power:
                    gpus_power[g] = list()
                if g not in data['capp_stack']:
                    gpus_power[g].append(0)
                else:    
                    gpu_power = data['wl_state'][GPUWorkloadType.BE][g]['cap']['current']
                    assert data['wl_state'][GPUWorkloadType.BE][g]['state'] == WorkloadState.RUNNING, f"GPU {g} found in cap_stack but not runnig stack:{data['wl_state'][GPUWorkloadType.BE][g]}"
                    gpus_power[g].append(gpu_power)

        for i, g in enumerate(self.workloads_state[GPUWorkloadType.BE]):
            gpu_np = np.array(gpus_power[g])
            accum = np.zeros_like(gpu_np)
            # add prev. gpus power to this as offset 
            for j in range(min(self.workloads_state[GPUWorkloadType.BE]), max(self.workloads_state[GPUWorkloadType.BE])+1):
                if j > g:
                    accum += np.array(gpus_power[j])
            plt.bar(np.array(x_ax_data)+0.5, gpu_np, width=1, bottom=accum, label=f"gpu{g}", ec='k')
        plt.step(np.array(x_ax_data)+0.5, targets, where='mid', label='Target', c='red')
        plt.legend(loc='upper left', ncol=3)
        # Save in png and pdf in provided log_dir
        plt.xlabel("step")
        plt.ylabel("Power (watt)")
        plt.ylim([0,max(targets)+70])
        plt.savefig(f"{save_path}/snap_shot.png")
        plt.savefig(f"{save_path}/snap_shot.pdf")
        plt.close()

    def run_lc(self, lc_trace: str):
        if self.debug:
            return
        with concurrent.futures.ThreadPoolExecutor() as executor:
            self.lc_thread = executor.submit(
                self.workload_runner.run_lc_client, 
                args=(False, 4, 60, self.lc_num_gpus_needed, 98, lc_trace, 60, False)
            )
        
    def wait_lc(self):
        return self.lc_thread.result()
    
    def __del__(self):
        self.docker_runner.cleanup()
