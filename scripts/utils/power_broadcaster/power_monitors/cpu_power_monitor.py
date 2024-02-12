# cmd: turbostat -S -i 0.5 -q --hide Avg_MHz,Busy%,Bzy_MHz,TSC_MHz,IPC,IRQ,POLL,C1,C2,POLL%,C1%,C2%,CorWatt 
import threading
import subprocess
import multiprocessing
import time
import sys

class CPUPowerMonitor(threading.Thread):
    lock = threading.Lock()
    def __init__(self, sampling_interval_msec = 500):
        self.set_powers = dict()
        self.sampling_interval_sec = sampling_interval_msec / 1000.0
        self.queue = multiprocessing.Queue()
        self.runnig = False
        threading.Thread.__init__(self)

    def run(self):
        # Collects GPU power data (per gpu) and calculates the average for the collected data
        with self.lock:
            self.runnig = True
        nb_samples = 1
        log = ""
        proc = subprocess.Popen(
            [
                'turbostat',
                '-S',
                '-i',
                f'{self.sampling_interval_sec}',
                '-q',
                '--hide',
                'Avg_MHz,Busy%,Bzy_MHz,TSC_MHz,IPC,IRQ,POLL,C1,C2,POLL%,C1%,C2%,CorWatt'
            ], 
            stdout=subprocess.PIPE,
            encoding='utf8'
        )
        while True:
            with self.lock:
                if self.runnig == False:
                    break
            power_t = self.get_cur_power(proc)
            if power_t is None: # just in case 
                continue
            power = int(power_t)
            if power == None:
                continue
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
                
                power = int(float(prev_powers[1] * prev_powers[0] + power) / float(prev_powers[0] +1))
                # prev_powers = self.queue.get()
            nb_samples += 1
                
            self.queue.put((nb_samples, power))
            time.sleep(self.sampling_interval_sec)
        f = open("cpu_power_log", 'w')
        f.write(log)
        f.close()
        proc.kill()

    def stop(self):
        running = False
        with self.lock:
            running = self.runnig
        if running:
            with self.lock:
                self.runnig = False
            self.join()
    
    def get_cur_power(self, proc):
        lines = list()
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if "PkgWatt" not in line and len(line) > 0:
                lines.append(line)
                break
        if len(lines) == 0:
            return None

        avg_pow = 0
        for p in lines:
            avg_pow += float(p)
        avg_pow /= len(lines)
        return avg_pow

    def get_avg_power(self):
        # Multiprocess queue communication for getting power data 
        return self.queue.get()[1]

    def __del__(self):
        self.stop()


if __name__ == "__main__":
    samples = 10
    interval_msec = 1000
    gmon = CPUPowerMonitor(interval_msec)
    gmon.start()
    print(f"Reading {samples} in {interval_msec}msec intervals")
    printed_samples = 1
    while printed_samples <= samples:
        print(f"#{printed_samples}:\t{gmon.get_avg_power()}")
        printed_samples += 1
        time.sleep(1)
    gmon.stop()
