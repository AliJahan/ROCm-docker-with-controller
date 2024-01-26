# cmd: turbostat -S -i 0.5 -q --hide Avg_MHz,Busy%,Bzy_MHz,TSC_MHz,IPC,IRQ,POLL,C1,C2,POLL%,C1%,C2%,CorWatt 
import sys
import threading
import subprocess
import multiprocessing
import time
import datetime 
import zmq
import sys

class CPUPowerMonitor(threading.Thread):
    num_gpus = 0
    lock = threading.Lock()
    def __init__(self, sampling_interval_msec = 250):
        self.set_powers = dict()
        self.sampling_interval = sampling_interval_msec
        self.queue = multiprocessing.Queue()
        self.runnig = False
        threading.Thread.__init__(self)

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
            power = int(self.get_cur_power(i))
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