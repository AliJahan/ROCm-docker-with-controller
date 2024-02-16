import os
import time
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # for utils TODO: make it a wheel

from utils.stat_collectors.power_collector import PowerCollector
from utils.workload_runners.workload_runner_remote import RemoteDockerRunner, RemoteWorkloadRunner
from utils.resource_managers.resource_manager import GPUResourceManager

class RemoteExperimentRunner:
    power_collector = None
    docker_runner = None
    workload_runner = None
    resource_controller = None
    dockers_to_cleanup = list()
    def __init__(
            self,
            lc_load_trace: str = "",
            target_ip: str = "172.20.0.9",
            remote_ip: str = "172.20.0.6",
            remote_resource_ctl_port: str = "5000",
            remote_workload_ctl_port: str = "3000",
            target_docker_ctrl_port: str = "4000",
            target_power_broadcaster_port: str = "6000",
            
        ):
        self.RESULTS_DIR = "/workspace/results"
        self.LOGS_DIR = "/workspace/experiment_logs"
        self.target_ip = target_ip
        self.remote_ip = remote_ip
        self.lc_load_trace = lc_load_trace
        self.remote_resource_ctl_port = remote_resource_ctl_port
        self.remote_workload_ctl_port = remote_workload_ctl_port
        self.target_docker_ctrl_port = target_docker_ctrl_port
        self.target_power_broadcaster_port = target_power_broadcaster_port

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
            remote_workload_control_port=self.remote_workload_ctl_port
        )
        
        self.resource_controller = GPUResourceManager(
            remote_control_ip=remote_ip,
            remote_control_port=remote_resource_ctl_port
        )

    def cleanup(self):
        print("Cleaning up the experiment")
        if self.power_collector is not None:
            del self.power_collector
        self.resource_controller.cleanup()
        self.docker_runner.cleanup()

    def setup_server(self):
        print(f"Staring power-broadcaster image on target server: {self.target_ip} ", flush=True)
        assert self.docker_runner.start_docker("power-broadcaster") == True, "Failed! stopping experiments!"
    def start_power_collection(self):
        assert self.power_collector is None, "power collector is alreary running"
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


    # Sweeps gpus for LC and collects qos and power data
    def run_lc_sweep_gpu_experiment(self, lc_load_steps: int, rounds: int, num_gpus: int, model: str = "resnet152"):
        args = dict()
        args['model'] = model
        args['batch_size'] = "1"
        expr_num_warmpup_load_steps=4
        expr_warmup_step_duration_sec=60
        expr_max_rps_per_gpu=100
        expr_trace_unit_sec=60
        def read_performed_expers(file_name):
            data = list()
            if os.path.isfile(file_name) == False:
                return data
            
            round = str(file_name.split("round")[-1])
            f = open(file_name, 'r')
            lines = f.readlines()
            f.close()
            
            for line in lines[1:]:
                line_d = line.split(",")
                data.append((int(round), int(line_d[0]), int(line_d[1])))
            print(f"Found {len(data)} experiments from previous run to skip!", flush=True)
            return data
        def save_results(gpu, load_pct, load_rps, powers, qos, results_file, create = False):
            if os.path.isfile(results_file) and create:
                create = False
            fd = open(results_file, 'w' if create else 'a')
            # 'num_samples': 480, 'sample_interval': 1
            cpu_avg = 0.0
            gpu_avg = dict()
            for power in powers['powers']:
                cpu_pow = power['cpu']
                gpus_pow = power['gpu']
                cpu_avg += float(cpu_pow)
                for g in gpus_pow:
                    if g not in gpu_avg:
                        gpu_avg[g] = 0
                    gpu_avg[g] += float(gpus_pow[g])
            cpu_avg /= float(powers['num_samples'])
            for g in gpu_avg:
                gpu_avg[g] /= float(powers['num_samples'])
            
            line = ""
            # header
            if create:
                line = "gpu,load_pct,load_rps,cpu_avg_pow,"
                for g in gpu_avg:
                    line += f"gpu_{g}_avg_pow,"
                # 'min': 15.0, 'max': 335.0, 'mean': 50.0, 'p25th': 37.0, 'p50th': 54.0, 'p75th': 57.0, 'p90th': 67.0, 'p95th': 78.0, 'p99th': 103.0
                line += "cpu_gpu_pow_avg,min,p25th,p50th,mean,p75th,p90th,p95th,p99th,max\n"
            # values
            # 1- gpus
            # 2- loads
            # 3- powers
            # 4- qos (if any)
                
            # 1,2,3(cpu power)
            line += f"{gpu},{load_pct},{load_rps},{int(cpu_avg)},"
            # 3 (gpus power)
            for g in gpu_avg:
                avg = int(gpu_avg[g])
                line += f"{avg},"
            # 3 (server power)
            cpu_gpu = int(cpu_avg) + int(gpu_avg['total'])
            line += f'{cpu_gpu},'
            # 4 
            if qos is None:
                line += "0,0,0,0,0,0,0,0,0\n"
            else:
                line += str(int(qos['min'])) + "," + str(int(qos['p25th'])) + "," + str(int(qos['p50th'])) + ","
                line += str(int(qos['mean'])) + "," + str(int(qos['p75th'])) + "," + str(int(qos['p90th'])) + ","
                line += str(int(qos['p95th'])) + "," + str(int(qos['p99th'])) + "," + str(int(qos['max'])) + "\n"
            # save into file
            fd.write(line)
            fd.close()
        # init
        trace_files_map = dict()
        
        print(f"Creating LC load trace file with {lc_load_steps}% steps ...", flush=True)
        
        for load_pct in range(lc_load_steps, 101, lc_load_steps):
            file_name = f'{self.LOGS_DIR}/lc_trace_file_load{load_pct}pct_steps{lc_load_steps}pct'
            trace_file = open(file_name, 'w')
            trace_file.write(f"{load_pct}\n")
            trace_file.close()
            trace_files_map[load_pct] = file_name
        
        print("Setting up target server ... ")
        
        for round in range(1, rounds+1):
            self.setup_server()
            results_file = f"{self.RESULTS_DIR}/lc_sweep_gpus{num_gpus}_lstep{lc_load_steps}_maxloadrps{expr_max_rps_per_gpu}_round{round}"
            prev_expr = read_performed_expers(results_file)
            if (int(round), int(0), int(0)) not in  prev_expr:
                print("Collecting idle powers for 30 sec ... ", end="", flush=True)
                self.start_power_collection()
                time.sleep(30)
                idle_powers = self.stop_power_collection()
                save_results(
                    gpu=0,
                    load_pct=0,
                    load_rps=0,
                    powers=idle_powers,qos=None,
                    results_file=results_file,
                    create=True
                )
                print("Done!", flush=True)
               
            gpu = num_gpus
            while gpu < num_gpus+1:
                print(f"Profiling power and QoS of Inference server with gpus:{gpu}" , flush=True)
                trace_files_map_currected = dict()
                for load_pct in trace_files_map:
                    if (int(round), int(gpu), int(load_pct)) in prev_expr:
                        print(f"\t-Skipping experiment for round:{round} gpu:{gpu} load:{load_pct}", flush=True)
                        continue
                    trace_files_map_currected[load_pct] = trace_files_map[load_pct]
                if len(list(trace_files_map_currected.keys())) == 0:
                    print(f"@@All experiments for round {round} are skipped, going to the next round", flush=True)
                    break
                print("1-Starting Inference-Server service ... " , flush=True)
                assert self.docker_runner.start_docker("Inference-Server")== True, f"Could not start Inference-Server service for gpu: {gpu}"
                print("2-Starting inference server... " , flush=True)
                assert self.workload_runner.start(is_be_wl=False), f"Could not start inference server for gpu: {gpu}"
                for g in range(0, gpu):
                    args['gpu'] = str(g)
                    print(f"\t2.1- Adding gpu {g}..." , flush=True)
                    assert self.workload_runner.add_gpu(is_be_wl=False, args=args), f"Could not add gpu {g} to inference server"
                # Warmup
                # return
                warmup_trace = trace_files_map_currected[list(trace_files_map_currected.keys())[0]]
                print(f"\t2.2-warming up with trace: {warmup_trace}" , flush=True)
                self.workload_runner.run_lc_client(
                    warmp_first=True,
                    num_warmpup_load_steps=expr_num_warmpup_load_steps,
                    warmup_step_duration_sec=expr_warmup_step_duration_sec,
                    gpus=gpu,
                    max_rps_per_gpu=expr_max_rps_per_gpu,
                    trace_file=warmup_trace,
                    trace_unit_sec=expr_trace_unit_sec
                )
                FAILED = False
                for load_pct in trace_files_map_currected:
                    print(f"3-Running client with load {load_pct}% trace:{trace_files_map_currected[load_pct]}" , flush=True)
                    self.start_power_collection()
                    qos_data = self.workload_runner.run_lc_client(
                        warmp_first=False,
                        num_warmpup_load_steps=expr_num_warmpup_load_steps,
                        warmup_step_duration_sec=expr_warmup_step_duration_sec,
                        gpus=gpu,
                        max_rps_per_gpu=expr_max_rps_per_gpu,
                        trace_file=trace_files_map_currected[load_pct],
                        trace_unit_sec=expr_trace_unit_sec
                    )
                    lc_powers = self.stop_power_collection()
                    if qos_data is None:
                        print(f"Error happened for running client for trace:{load_pct}%. Restarting..." , flush=True)    
                        FAILED = True
                        break
                    print(f"3-Saving power and qos for load {load_pct}%" , flush=True)

                    rps = int((float(load_pct) / 100.0) * num_gpus * expr_max_rps_per_gpu)
                    save_results(
                        gpu=gpu,
                        load_pct=load_pct,
                        load_rps=rps,
                        powers=lc_powers,
                        qos=qos_data,
                        results_file=results_file,
                        create=False
                    )
                    print(f"---------------------------------------" , flush=True)
                print("4-Stopping Inference-Server service ... " , flush=True)
                assert self.docker_runner.stop_docker("Inference-Server")== True, "Could not stop Inference-Servre service"
                if FAILED == False:
                    gpu += 1
            self.cleanup()

    def run_sweep_lc_load_vs_power_cap(self, lc_load_steps: int, power_cap_step: int, rounds: int, num_gpus: int, model: str = "resnet152"):
        args = dict()
        args['model'] = model
        args['batch_size'] = "1"
        expr_num_warmpup_load_steps=4
        expr_warmup_step_duration_sec=60
        expr_max_rps_per_gpu=100
        expr_trace_unit_sec=60
        def read_performed_expers(file_name, round):
            data = list()
            if os.path.isfile(file_name) == False:
                return data
            
            # round = str(file_name.split("round")[-1])
            f = open(file_name, 'r')
            lines = f.readlines()
            f.close()
            
            for line in lines[1:]:
                line_d = line.split(",")
                # (round, gpu, cap, load)
                data.append((int(round), int(line_d[0]), int(line_d[1]), int(line_d[2])))
            print(f"Found {len(data)} experiments from previous run to skip!", flush=True)
            return data
        def save_results(gpu, load_pct, load_rps, cap, powers, qos, results_file, create = False):
            if os.path.isfile(results_file) and create:
                create = False
            fd = open(results_file, 'w' if create else 'a')
            # 'num_samples': 480, 'sample_interval': 1
            cpu_avg = 0.0
            gpu_avg = dict()
            for power in powers['powers']:
                cpu_pow = power['cpu']
                gpus_pow = power['gpu']
                cpu_avg += float(cpu_pow)
                for g in gpus_pow:
                    if g not in gpu_avg:
                        gpu_avg[g] = 0
                    gpu_avg[g] += float(gpus_pow[g])
            cpu_avg /= float(powers['num_samples'])
            for g in gpu_avg:
                gpu_avg[g] /= float(powers['num_samples'])
            
            line = ""
            # header
            if create:
                line = "gpu,cap,load_pct,load_rps,cpu_avg_pow,"
                for g in gpu_avg:
                    line += f"gpu_{g}_avg_pow,"
                # 'min': 15.0, 'max': 335.0, 'mean': 50.0, 'p25th': 37.0, 'p50th': 54.0, 'p75th': 57.0, 'p90th': 67.0, 'p95th': 78.0, 'p99th': 103.0
                line += "cpu_gpu_pow_avg,min,p25th,p50th,mean,p75th,p90th,p95th,p99th,max\n"
            # values
            # 1- gpus
            # 2- loads
            # 3- powers
            # 4- qos (if any)
                
            # 1,2,3(cpu power)
            line += f"{gpu},{cap},{load_pct},{load_rps},{int(cpu_avg)},"
            # 3 (gpus power)
            for g in gpu_avg:
                avg = int(gpu_avg[g])
                line += f"{avg},"
            # 3 (server power)
            cpu_gpu = int(cpu_avg) + int(gpu_avg['total'])
            line += f'{cpu_gpu},'
            # 4 
            if qos is None:
                line += "0,0,0,0,0,0,0,0,0\n"
            else:
                line += str(int(qos['min'])) + "," + str(int(qos['p25th'])) + "," + str(int(qos['p50th'])) + ","
                line += str(int(qos['mean'])) + "," + str(int(qos['p75th'])) + "," + str(int(qos['p90th'])) + ","
                line += str(int(qos['p95th'])) + "," + str(int(qos['p99th'])) + "," + str(int(qos['max'])) + "\n"
            # save into file
            fd.write(line)
            fd.close()
        # init
        trace_files_map = dict()
        
        print(f"Creating LC load trace file with {lc_load_steps}% steps ...", flush=True)
        
        for load_pct in range(lc_load_steps, 101, lc_load_steps):
            file_name = f'{self.LOGS_DIR}/lc_trace_file_load{load_pct}pct_steps{lc_load_steps}pct'
            trace_file = open(file_name, 'w')
            trace_file.write(f"{load_pct}\n")
            trace_file.close()
            trace_files_map[load_pct] = file_name
        
        print("Setting up target server ... ")
        
        for round in range(1, rounds+1): 
            gpu = num_gpus
            for power_cap in range(225, 0,-power_cap_step):
                results_file = f"{self.RESULTS_DIR}/lc_load_vs_cap{power_cap}_gpus{num_gpus}_lstep{lc_load_steps}_capstep{power_cap_step}_maxloadrps{expr_max_rps_per_gpu}_round{round}"
                # skip performed experiments
                prev_expr = read_performed_expers(file_name=results_file, round=round)
                trace_files_map_currected = dict()
                for load_pct in trace_files_map:
                    if (int(round), int(gpu), int(power_cap), int(load_pct)) in prev_expr:
                        print(f"\t-Skipping experiment for round:{round} gpu:{gpu} power_cap:{power_cap} load:{load_pct}", flush=True)
                        continue
                    trace_files_map_currected[load_pct] = trace_files_map[load_pct]
                if len(list(trace_files_map_currected.keys())) == 0:
                    print(f"@@All experiments for round {round} are skipped, going to the power_cap", flush=True)
                    continue
                # there is at list one experiment for this power_cap
                print("Experiment begin-----------------------------" , flush=True)
                print("Setting up power-broadvaster on target server" , flush=True)
                print("-------------" , flush=True)
                # self.setup_server()
                if (int(round), int(gpu), int(power_cap), int(0)) not in  prev_expr:
                    assert self.docker_runner.start_docker("power-broadcaster") == True, "Failed! stopping experiments!"
                    print("Collecting idle powers for 30 sec ... ", end="", flush=True)
                    self.start_power_collection()
                    time.sleep(30)
                    idle_powers = self.stop_power_collection()
                    save_results(
                        gpu=gpu,
                        load_pct=0,
                        load_rps=0,
                        cap=power_cap,
                        powers=idle_powers,qos=None,
                        results_file=results_file,
                        create=True
                    )
                    print("Done!", flush=True)
                    assert self.docker_runner.stop_docker("power-broadcaster") == True, "Failed! stopping experiments!"


                print(f"Profiling power and QoS of Inference server with gpus:{gpu}" , flush=True)
                print("1-Starting Inference-Server service ... " , flush=True)
                assert self.docker_runner.start_docker("Inference-Server")== True, f"Could not start Inference-Server service for gpu: {gpu}"
                print("2-Starting inference server... " , flush=True)
                assert self.workload_runner.start(is_be_wl=False), f"Could not start inference server for gpu: {gpu}"
                for g in range(0, gpu):
                    args['gpu'] = str(g)
                    print(f"\t2.1- Adding gpu {g}..." , flush=True)
                    assert self.workload_runner.add_gpu(is_be_wl=False, args=args), f"Could not add gpu {g} to inference server"
                    print(f"\t2.2- Setting power cap for gpu {g} to 225 (reset)..." , flush=True)
                    assert self.resource_controller.set_freq(app_name="Inference-Server", gpu=g, freq=225), f"Could not set power cap to 225 for gpu {g}"
                # Warmup
                # return
                warmup_trace = trace_files_map_currected[list(trace_files_map_currected.keys())[0]]
                print(f"\t2.3-warming up with trace: {warmup_trace}" , flush=True)
                self.workload_runner.run_lc_client(
                    warmp_first=True,
                    num_warmpup_load_steps=expr_num_warmpup_load_steps,
                    warmup_step_duration_sec=expr_warmup_step_duration_sec,
                    gpus=gpu,
                    max_rps_per_gpu=expr_max_rps_per_gpu,
                    trace_file=warmup_trace,
                    trace_unit_sec=expr_trace_unit_sec
                )
                for g in range(0, gpu):
                    print(f"\t2.4- Setting power cap for gpu {g} to {power_cap} (reset)..." , flush=True)
                    assert self.resource_controller.set_freq(app_name="Inference-Server", gpu=g, freq=power_cap), f"Could not set power cap to {power_cap} for gpu {g}"
                for load_pct in trace_files_map_currected:
                    assert self.docker_runner.start_docker("power-broadcaster") == True, "Failed! stopping experiments!"
                    print(f"3-Running client with load {load_pct}% trace:{trace_files_map_currected[load_pct]}" , flush=True)
                    self.start_power_collection()
                    qos_data = self.workload_runner.run_lc_client(
                        warmp_first=False,
                        num_warmpup_load_steps=expr_num_warmpup_load_steps,
                        warmup_step_duration_sec=expr_warmup_step_duration_sec,
                        gpus=gpu,
                        max_rps_per_gpu=expr_max_rps_per_gpu,
                        trace_file=trace_files_map_currected[load_pct],
                        trace_unit_sec=expr_trace_unit_sec
                    )
                    lc_powers = self.stop_power_collection()
                    if qos_data is None:
                        print(f"Error happened for running client for trace:{load_pct}%. Restarting..." , flush=True)    
                        break
                    print(f"3-Saving power and qos for load {load_pct}%" , flush=True)

                    rps = int((float(load_pct) / 100.0) * num_gpus * expr_max_rps_per_gpu)
                    save_results(
                        gpu=gpu,
                        load_pct=load_pct,
                        load_rps=rps,
                        cap=power_cap,
                        powers=lc_powers,
                        qos=qos_data,
                        results_file=results_file,
                        create=False
                    )
                    assert self.docker_runner.stop_docker("power-broadcaster") == True, "Failed! stopping experiments!"
                    print(f"---------------------------------------" , flush=True)
                for g in range(0, gpu):
                    print(f"4-Resetting power cap for gpu {g} to 225 (reset)..." , flush=True)
                    assert self.resource_controller.set_freq(app_name="Inference-Server", gpu=g, freq=225), f"Could not set power cap to 225 for gpu {g}"
                print("5-Stopping Inference-Server service ... " , flush=True)
                assert self.docker_runner.stop_docker("Inference-Server")== True, "Could not stop Inference-Servre service"
                self.cleanup()
                print("Experiment end-------------------------------" , flush=True)
            
    def run_sweep_lc_load_vs_cu_mask(self, lc_load_steps: int, cumask_step: int, rounds: int, num_gpus: int, model: str = "resnet152"):
        args = dict()
        args['model'] = model
        args['batch_size'] = "1"
        expr_num_warmpup_load_steps=4
        expr_warmup_step_duration_sec=60
        expr_max_rps_per_gpu=100
        expr_trace_unit_sec=60
        def read_performed_expers(file_name, round):
            data = list()
            if os.path.isfile(file_name) == False:
                return data
            
            # round = str(file_name.split("round")[-1])
            f = open(file_name, 'r')
            lines = f.readlines()
            f.close()
            
            for line in lines[1:]:
                line_d = line.split(",")
                # (round, gpu, mask, load)
                data.append((int(round), int(line_d[0]), int(line_d[1]), int(line_d[2])))
            print(f"Found {len(data)} experiments from previous run to skip!", flush=True)
            return data
        def save_results(gpu, load_pct, load_rps, cus, powers, qos, results_file, create = False):
            if os.path.isfile(results_file) and create:
                create = False
            fd = open(results_file, 'w' if create else 'a')
            # 'num_samples': 480, 'sample_interval': 1
            cpu_avg = 0.0
            gpu_avg = dict()
            for power in powers['powers']:
                cpu_pow = power['cpu']
                gpus_pow = power['gpu']
                cpu_avg += float(cpu_pow)
                for g in gpus_pow:
                    if g not in gpu_avg:
                        gpu_avg[g] = 0
                    gpu_avg[g] += float(gpus_pow[g])
            cpu_avg /= float(powers['num_samples'])
            for g in gpu_avg:
                gpu_avg[g] /= float(powers['num_samples'])
            
            line = ""
            # header
            if create:
                line = "gpu,cus,load_pct,load_rps,cpu_avg_pow,"
                for g in gpu_avg:
                    line += f"gpu_{g}_avg_pow,"
                # 'min': 15.0, 'max': 335.0, 'mean': 50.0, 'p25th': 37.0, 'p50th': 54.0, 'p75th': 57.0, 'p90th': 67.0, 'p95th': 78.0, 'p99th': 103.0
                line += "cpu_gpu_pow_avg,min,p25th,p50th,mean,p75th,p90th,p95th,p99th,max\n"
            # values
            # 1- gpus
            # 2- loads
            # 3- powers
            # 4- qos (if any)
                
            # 1,2,3(cpu power)
            line += f"{gpu},{cus},{load_pct},{load_rps},{int(cpu_avg)},"
            # 3 (gpus power)
            for g in gpu_avg:
                avg = int(gpu_avg[g])
                line += f"{avg},"
            # 3 (server power)
            cpu_gpu = int(cpu_avg) + int(gpu_avg['total'])
            line += f'{cpu_gpu},'
            # 4 
            if qos is None:
                line += "0,0,0,0,0,0,0,0,0\n"
            else:
                line += str(int(qos['min'])) + "," + str(int(qos['p25th'])) + "," + str(int(qos['p50th'])) + ","
                line += str(int(qos['mean'])) + "," + str(int(qos['p75th'])) + "," + str(int(qos['p90th'])) + ","
                line += str(int(qos['p95th'])) + "," + str(int(qos['p99th'])) + "," + str(int(qos['max'])) + "\n"
            # save into file
            fd.write(line)
            fd.close()
        # init
        trace_files_map = dict()
        
        print(f"Creating LC load trace file with {lc_load_steps}% steps ...", flush=True)
        
        for load_pct in range(lc_load_steps, 101, lc_load_steps):
            file_name = f'{self.LOGS_DIR}/lc_trace_file_load{load_pct}pct_steps{lc_load_steps}pct'
            trace_file = open(file_name, 'w')
            trace_file.write(f"{load_pct}\n")
            trace_file.close()
            trace_files_map[load_pct] = file_name
        
        print("Setting up target server ... ")
        
        for round in range(1, rounds+1): 
            gpu = num_gpus
            for n_cus in range(60//cumask_step):
                total_cus = 60 - (n_cus)*cumask_step
                results_file = f"{self.RESULTS_DIR}/lc_load_vs_cus{total_cus}_gpus{num_gpus}_lstep{lc_load_steps}_custep{cumask_step}_maxloadrps{expr_max_rps_per_gpu}_round{round}"
                # skip performed experiments
                prev_expr = read_performed_expers(file_name=results_file, round=round)
                trace_files_map_currected = dict()
                for load_pct in trace_files_map:
                    if (int(round), int(gpu), int(total_cus), int(load_pct)) in prev_expr:
                        print(f"\t-Skipping experiment for round:{round} gpu:{gpu} num_cus:{total_cus} load:{load_pct}", flush=True)
                        continue
                    trace_files_map_currected[load_pct] = trace_files_map[load_pct]
                if len(list(trace_files_map_currected.keys())) == 0:
                    print(f"@@All experiments for round:{round} cus: {total_cus} gpus: {gpu} are skipped, going to the next round", flush=True)
                    continue
                # there is at list one experiment for this power_cap
                print(f"Experiment begin(for cus:{total_cus})-----------------" , flush=True)
                print("Setting up power-broadvaster on target server" , flush=True)
                print("-------------" , flush=True)
                # self.setup_server()
                if (int(round), int(gpu), int(total_cus), int(0)) not in  prev_expr:
                    assert self.docker_runner.start_docker("power-broadcaster") == True, "Failed! stopping experiments!"
                    print("Collecting idle powers for 30 sec ... ", end="", flush=True)
                    self.start_power_collection()
                    time.sleep(30)
                    idle_powers = self.stop_power_collection()
                    save_results(
                        gpu=gpu,
                        load_pct=0,
                        load_rps=0,
                        cus=total_cus,
                        powers=idle_powers,qos=None,
                        results_file=results_file,
                        create=True
                    )
                    assert self.docker_runner.stop_docker("power-broadcaster") == True, "Failed! stopping experiments!"
                    print("Done!", flush=True)
              
                print(f"Profiling power and QoS of Inference server with gpus:{gpu}" , flush=True)
                
                print("1-Starting Inference-Server service ... " , flush=True)
                assert self.docker_runner.start_docker("Inference-Server")== True, f"Could not start Inference-Server service for gpu: {gpu}"
                print("2-Starting inference server... " , flush=True)
                assert self.workload_runner.start(is_be_wl=False), f"Could not start inference server for gpu: {gpu}"
                for g in range(0, gpu):
                    args['gpu'] = str(g)
                    print(f"\t2.1- Adding gpu {g}..." , flush=True)
                    assert self.workload_runner.add_gpu(is_be_wl=False, args=args), f"Could not add gpu {g} to inference server"
                    print(f"\t2.2- Setting num cus for gpu {g} to 60 (reset)..." , flush=True)
                    current_cus = self.resource_controller.get_current_cus(gpu=g, is_be=False)                    
                    assert self.resource_controller.add_cu(app_name="Inference-Server", gpu=g, cus=(60-current_cus), is_be=False), f"Could not set cu mask to 60 for gpu {g}"
                # Warmup
                # return
                warmup_trace = trace_files_map_currected[list(trace_files_map_currected.keys())[0]]
                print(f"\t2.3-warming up with trace: {warmup_trace}" , flush=True)
                self.workload_runner.run_lc_client(
                    warmp_first=True,
                    num_warmpup_load_steps=expr_num_warmpup_load_steps,
                    warmup_step_duration_sec=expr_warmup_step_duration_sec,
                    gpus=gpu,
                    max_rps_per_gpu=expr_max_rps_per_gpu,
                    trace_file=warmup_trace,
                    trace_unit_sec=expr_trace_unit_sec
                )
                for g in range(0, gpu):
                    current_cus = self.resource_controller.get_current_cus(gpu=g, is_be=False)
                    print(f"\t2.4- Setting num cus for gpu {g} to from {current_cus} cus, removing {(n_cus)*cumask_step} (reset)..." , flush=True)
                    assert self.resource_controller.remove_cu(app_name="Inference-Server", gpu=g, cus=(current_cus)), f"Could not remove {current_cus} cus for gpu {g}"
                    assert self.resource_controller.add_cu(app_name="Inference-Server", gpu=g, cus=(total_cus)), f"Could not add {total_cus} cus for gpu {g}"
                for load_pct in trace_files_map_currected:
                    assert self.docker_runner.start_docker("power-broadcaster") == True, "Failed! stopping experiments!"
                    print(f"3-Running client with load {load_pct}% trace:{trace_files_map_currected[load_pct]}" , flush=True)
                    self.start_power_collection()
                    qos_data = self.workload_runner.run_lc_client(
                        warmp_first=False,
                        num_warmpup_load_steps=expr_num_warmpup_load_steps,
                        warmup_step_duration_sec=expr_warmup_step_duration_sec,
                        gpus=gpu,
                        max_rps_per_gpu=expr_max_rps_per_gpu,
                        trace_file=trace_files_map_currected[load_pct],
                        trace_unit_sec=expr_trace_unit_sec
                    )
                    lc_powers = self.stop_power_collection()
                    if qos_data is None:
                        print(f"Error happened for running client for trace:{load_pct}%. Restarting..." , flush=True)    
                        break
                    print(f"3-Saving power and qos for load {load_pct}%" , flush=True)

                    rps = int((float(load_pct) / 100.0) * num_gpus * expr_max_rps_per_gpu)
                    save_results(
                        gpu=gpu,
                        load_pct=load_pct,
                        load_rps=rps,
                        cus=total_cus,
                        powers=lc_powers,
                        qos=qos_data,
                        results_file=results_file,
                        create=False
                    )
                    print(f"4-Stopping power-broadcaster for {load_pct}%" , flush=True)
                    assert self.docker_runner.stop_docker("power-broadcaster") == True, "Failed! stopping experiments!"
                    print(f"---------------------------------------" , flush=True)
                for g in range(0, gpu):
                    current_cus = self.resource_controller.get_current_cus(gpu=gpu, is_be=False)
                    print(f"4.1-Removing {current_cus} cus from gpu {g} ... " , flush=True)
                    assert self.resource_controller.remove_cu(app_name="Inference-Server", gpu=g, cus=current_cus), f"Could not remove {current_cus} cus for gpu {g}"
                print("5-Stopping Inference-Server service ... " , flush=True)
                assert self.docker_runner.stop_docker("Inference-Server")== True, "Could not stop Inference-Servre service"
                self.cleanup()
                print("Experiment end-------------------------------" , flush=True)
    def __del__(self):
        self.cleanup()

if __name__ == "__main__":
    remote_runner = RemoteExperimentRunner()

    # for gpu in [1]: ## get max rps per various number of GPUs
    #     remote_runner.run_lc_sweep_gpu_experiment(lc_load_steps=2, rounds=2, num_gpus=gpu)
    # for gpu in [1]: # power cap vs qos
    #     remote_runner.run_sweep_lc_load_vs_power_cap(lc_load_steps=2, power_cap_step=5, rounds=1, num_gpus=gpu)
    for gpu in [1]: # cu vs qos
        remote_runner.run_sweep_lc_load_vs_cu_mask(lc_load_steps=2, cumask_step=2, rounds=1, num_gpus=gpu)
    
    print("Done with experiment waiting 10 sec")
    # time.sleep(10000)

#IP=`ip r | head -n 1 | awk n 1 | awk '{p`intf $3}'