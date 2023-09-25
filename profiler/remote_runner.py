import sys
import threading
import subprocess
import multiprocessing
import time
import datetime 
# import rocm-smi lib
sys.path.append('/opt/rocm//libexec/rocm_smi/')
from rsmiBindings import *
import zmq
import sys

class GPUController(threading.Thread):
    num_gpus = 0
    lock = threading.Lock()
    def __init__(self, sampling_interval_msec = 250):
        if self.initializeRsmi() is False:
            print("ROCM SMI can not be initted\n", flush=True)
            sys.exit(0)
        self.set_powers = dict()
        self.sampling_interval = sampling_interval_msec
        self.queue = multiprocessing.Queue()
        self.runnig = False
        # make sure all gpus power is max
        for dev in range(self.num_gpus):
            self.set_power(dev, 225)
        threading.Thread.__init__(self)

    def driverInitialized(self):
        """ Returns true if amdgpu is found in the list of initialized modules
        """
        driverInitialized = ''
        try:
            driverInitialized = str(subprocess.check_output("cat /sys/module/amdgpu/initstate |grep live", shell=True))
        except subprocess.CalledProcessError:
            return False

        if len(driverInitialized) > 0:
            return True
        return False
    def rsmi_ret_ok(self, my_ret, device=None, metric=None, silent=False):
        """ Returns true if RSMI call status is 0 (success)

        If status is not 0, error logs are written to the debug log and false is returned

        @param device: DRM device identifier
        @param my_ret: Return of RSMI call (rocm_smi_lib API)
        @param metric: Parameter of GPU currently being analyzed
        """
        global RETCODE
        global PRINT_JSON
        if my_ret != rsmi_status_t.RSMI_STATUS_SUCCESS:
            err_str = c_char_p()
            rocmsmi.rsmi_status_string(my_ret, byref(err_str))
            returnString = ''
            if device is not None:
                returnString += '%s GPU[%s]:' % (my_ret, device)
            if metric is not None:
                returnString += ' %s: ' % (metric)
            returnString += '%s\t' % (err_str.value.decode())
            if not PRINT_JSON:
                if silent:
                    logging.info('%s', returnString)
                else:
                    logging.error('%s', returnString)
            RETCODE = my_ret
            return False
        return True
    def initializeRsmi(self):
        """ initializes rocmsmi if the amdgpu driver is initialized
        """
        # Check if amdgpu is initialized before initializing rsmi
        if self.driverInitialized() is True:
            ret_init = rocmsmi.rsmi_init(0)
            if ret_init != 0:
                print(f'ROCm SMI returned {ret_init} (the expected value is 0)')
                exit(ret_init)
            numberOfDevices = c_uint32(0)
            ret = rocmsmi.rsmi_num_monitor_devices(byref(numberOfDevices))
            if self.rsmi_ret_ok(ret):
                with self.lock:
                    self.num_gpus = numberOfDevices.value
            else:
                exit(ret)
        else:
            print('Driver not initialized (amdgpu not found in modules)')
            exit(0)
        return True

    def run(self):
        # Collects GPU power data (per gpu) and calculates the average for the collected data
        with self.lock:
            self.runnig = True
        nb_samples = 1
        log = ""
        while True:
            with self.lock:
                if self.runnig == False:
                    break
            power = dict()
            for i in range(self.num_gpus):
                power[i] = int(self.get_cur_power(i))
            # calc. avg. and update    
            if self.queue.empty(): # no avg for the first sample
                nb_samples = 1
                self.queue.put((1, power))
            else:
                prev_powers = self.queue.get()
                log += str(prev_powers) + "\n"
                # prev_powers = self.queue.put(prev_powers)
                # calc. avg.
                # print(f"{(prev_powers[1] * prev_powers[0] + power) / (prev_powers[0] +1 )} = {prev_powers[1]} * {prev_powers[0]} + {power} / ({prev_powers[0]} +1) ", flush=True)
                for i in range(self.num_gpus):
                    power[i] = int(float(prev_powers[1][i] * prev_powers[0] + power[i]) / float(prev_powers[0] +1))
                # prev_powers = self.queue.get()
            nb_samples += 1
                
            self.queue.put((nb_samples, power))
            time.sleep(self.sampling_interval / 1000.0)
        f = open("power", 'w')
        f.write(log)
        f.close()

    def stop(self):
        running = False
        with self.lock:
            running = self.runnig
        if running:
            with self.lock:
                self.runnig = False
            self.join()

    def get_num_gpus(self):
        gpus = 0
        with self.lock:
            gpus = self.num_gpus
        return gpus

    def get_cur_power_list(self, devs_lst):
        powers = 0
        for dev in devs_lst:
            power = c_uint32()
            ret = rocmsmi.rsmi_dev_power_ave_get(dev, 0, byref(power))
            powers += power.value / 1000000
        return powers
    
    def get_cur_power(self, dev):
        power = c_uint32()
        ret = rocmsmi.rsmi_dev_power_ave_get(dev, 0, byref(power))
        return power.value / 1000000
    
    def get_cur_power_all(self):
        power = 0
        for i in range(self.num_gpus):
            power += self.get_cur_power(i)
        return power
    def get_avg_power(self):
        # Multiprocess queue communication for getting power data 
        return self.queue.get()[1]

    def set_power(self, dev, power):
        power_cap = c_uint64()
        power_cap.value = int(power) * 1000000
        rocmsmi.rsmi_dev_power_cap_set(dev, 0, power_cap)
        if power != 225:
            self.set_powers[dev] = power

    def __del__(self):
        self.stop()
        for dev in self.set_powers:
            self.set_power(dev, 225)
        rocmsmi.rsmi_shut_down()

# master in this context is the binary which actually performs power capping and cu masking through ZMQ
# master connects to controller process in docker (through) ZMQ and asks for cu masking or power cap
MASTER_PATH="/home/ajaha004/repos/rocr/standalone-docker/ROCm-docker-with-controller/profiler/build/bin/master"

# During this time, the benchmark is running and power metrics are collected
EXPR_DURATION_SEC=120 

# During this time, the local_runner has started the process already by "start command", 
#    we wait a bit before power stat collection to get more stable results 
#    in other words, pausing power collection during benchmark setup and warmup.
START_WAIT=30 

class ProfilerController:
    def __init__(self, port: str = "5951"):
        # setup ZMQ for communication
        context = zmq.Context()
        self.sock_ = context.socket(zmq.DEALER)
        try:
            self.sock_.connect("tcp://localhost:"+port)
        except:
            print(f"Could not bind to tcp://*:{port}")
        # threading.Thread.__init__(self)
    def now_str(self):
        # returns current time as str
        return datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    def profile(self):
        # Profiling space
        cus = [i for i in range(60, 0, -2)]
        powers = [i for i in range(225, 0, -5)]
        # Some experiment info: for example, duration in hours
        dur = ((EXPR_DURATION_SEC + START_WAIT) / 60) * len(powers) * len(cus) / 60
        print(f"Total experiments: {len(cus)*len(powers)} ~Duration: {dur} hrs", flush=True)
        # csv log header
        log = "expr_ind,set_cu,set_power,avg_power,avg_tp"
        expr_ind = 1
        # Make sure the local runner is ready, we wait here for a response from local_runner before starting profiling
        self.sock_.send_string("READY")
        rep = self.sock_.recv_string()
        if "fail" in rep:
            print("Could not connect to local_runner, stopped")
            return
        # Profiling
        for power in powers:
            for cu in cus:
                print(f"@{self.now_str()}:{expr_ind}: starting experiment for: cu({cu}) power({power})", flush=True)
                # setup cu mask and power communicate with controller service
                set_cu_cmd = (MASTER_PATH + " cu " + str(cu)).split(" ")
                set_power_cmd = (MASTER_PATH + " power " + str(power)).split(" ")
                cu_set_p = subprocess.Popen(
                    set_cu_cmd,
                    stdout=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                time.sleep(1) 
                power_set_p = subprocess.Popen(
                    set_power_cmd,
                    stdout=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                # wait for cu and power setting to finish
                cu_set_p.communicate()
                power_set_p.communicate()
                print(f"\t- cu and power set", flush=True)
                print(f"\t- starting benchmark remotely", flush=True)
                # REMOTELY start benchmark (communicate with local_runner through ZMQ)
                self.sock_.send_string("START")
                rep = self.sock_.recv_string()
                print(f"\t- remote reply {rep}", flush=True)
                if "fail" in rep:
                    print("Experiment with power: {power} cu: {cu} failed")
                    return 
                print(f"\t- waiting for {START_WAIT} sec", flush=True)
                # Wait for started benchmark initialization/warmup
                time.sleep(START_WAIT)
                # Start power stat collection 
                print(f"\t- started power monitor, collecting power data for {EXPR_DURATION_SEC} sec", flush=True)
                gpu_reader = GPUController()
                gpu_reader.start()
                # Wait
                time.sleep(EXPR_DURATION_SEC)
                gpu_reader.stop()
                recorder = gpu_reader.get_avg_power()
                del gpu_reader
                print(f"\t- stopping benchmark remotely", flush=True)
                # STOP remote benchmark
                self.sock_.send_string("STOP")
                avg_tp = self.sock_.recv_string()
                if "fail" in avg_tp:
                    print(f"Experiment with power: {power} cu: {cu} failed with error {avg_tp}")
                    return
                print(f"\t- remote stopped BE", flush=True)
                # collect stats and save them to file
                print(f"\t- gpus power {recorder}", flush=True)
                new_log_entry = f"@{self.now_str()}: expr_ind: {expr_ind} cu:{cu} power: {power} power: {recorder[0]} avg_tp: {avg_tp}"
                new_rep = f"{expr_ind},{cu},{power},{recorder[0]},{avg_tp}\n"
                log += new_rep
                expr_ind += 1
                print(new_log_entry, flush=True)
                f = open("profiling_data", 'a')
                f.write(new_rep)
                f.close()

    
if __name__ == "__main__":
    profiler = ProfilerController()
    profiler.profile()
