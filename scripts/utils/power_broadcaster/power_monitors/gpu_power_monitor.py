import sys
import threading
import subprocess
import multiprocessing
import time
import datetime 
sys.path.append('/opt/rocm/libexec/rocm_smi/')
from rsmiBindings import *
import sys

class GPUPowerMonitor(threading.Thread):
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
        # for dev in range(self.num_gpus):
        #     self.set_power(dev, 225)
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
                return False
        else:
            print('Driver not initialized (amdgpu not found in modules)')
            return False

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
                power[str(i)] = int(self.get_cur_power(i))
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
                    power[str(i)] = int(float(prev_powers[1][str(i)] * prev_powers[0] + power[str(i)]) / float(prev_powers[0] +1))
                # prev_powers = self.queue.get()
            nb_samples += 1
                
            self.queue.put((nb_samples, power))
            time.sleep(self.sampling_interval / 1000.0)
        f = open("gpu_power_log", 'w')
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
        powers = self.queue.get()[1]
        total = 0
        for i in powers:
            total += powers[i]
        powers['total'] = total
        return powers

    # def set_power(self, dev, power):
    #     power_cap = c_uint64()
    #     power_cap.value = int(power) * 1000000
    #     rocmsmi.rsmi_dev_power_cap_set(dev, 0, power_cap)
    #     if power != 225:
    #         self.set_powers[dev] = power

    def __del__(self):
        self.stop()
        # for dev in self.set_powers:
        #     self.set_power(dev, 225)
        rocmsmi.rsmi_shut_down()

if __name__ == "__main__":
    samples = 10
    interval_msec = 1000
    gmon = GPUPowerMonitor(interval_msec)
    gmon.start()
    print(f"Reading {samples} in {interval_msec}msec intervals")
    printed_samples = 1
    while printed_samples <= samples:
        print(f"#{printed_samples}:\t{gmon.get_avg_power()}")
        printed_samples += 1
    gmon.stop()
