import zmq
import sys
import time
import json
import threading
import multiprocessing
class PowerCollector(threading.Thread):
    lock = threading.Lock()
    queue = multiprocessing.Queue()
    power_socket = None
    poller = None
    def __init__(self, collection_interval_sec: int = 1, power_broadcaster_port: str = "6000"):
        self.collection_interval_sec = collection_interval_sec
        self.power_broadcaster_port = power_broadcaster_port
        self.initialized = self.init()
        self.runnig = False
        self.num_samples = 0
        self.powers_list = list()
        threading.Thread.__init__(self)
    def is_initted(self):
        return self.initialized
    
    def init(self):
        self.power_socket, self.poller = self.setup_socket(self.power_broadcaster_port)
        if  self.power_socket is None:
            return False
    def deinit(self):
        if self.power_socket is not None:
            del self.power_socket
        self.ctx.term()
        

    def setup_socket(self, port):
        print(f"Creating power collection socket on port: {port}")
        self.ctx = zmq.Context.instance()
        publisher = self.ctx.socket(zmq.SUB)
        poller = None
        print(f"Connecting ... ", end="")
        try:
            publisher.connect(f"tcp://localhost:{port}")
            publisher.setsockopt(zmq.SUBSCRIBE, b"")
            publisher.setsockopt(zmq.CONFLATE, 1)
            poller = zmq.Poller()
            poller.register(publisher, zmq.POLLIN)
            print("Success!")
            return publisher, poller
        except Exception as e:
            print(f"Failed! error: {e}")
        return None, None

    def run(self):
        with self.lock:
            self.runnig = True
        print("PowerCollector is running...")
        while True:
            socks = dict(self.poller.poll(5))

            if self.power_socket in socks and socks[self.power_socket] == zmq.POLLIN:
                msg = None
                try:
                    msg = self.power_socket.recv_string()
                except zmq.ZMQError as e:
                    if e.errno == zmq.ETERM:
                        print("ZMQ socket interrupted/terminated, Quitting...")
                    else:
                        print(f"ZMQ socket error: {e}, Quitting...")
                    break
                # It should not be None here (just in case)
                if msg is None:
                    continue
                # Record current power data
                rcvd_power_data = json.loads(msg)
                with self.lock:
                    self.powers_list.append(rcvd_power_data)
                    self.num_samples += 1
                if self.queue.empty() == False:
                    self.queue.get()
                self.queue.put(rcvd_power_data)
            with self.lock:
                if self.runnig == False:
                    break
            # time.sleep(self.collection_interval_sec)

    def get_cur_power(self): # gets the last power published
        return self.queue.get()

    def get_all_powers(self):
        powers = list()
        num_samples = 0
        with self.lock:
            powers = self.powers_list
            num_samples = self.num_samples
        return {"powers": powers, "num_samples": num_samples, "sample_interval": self.collection_interval_sec}
    def stop(self):
        running = False
        with self.lock:
            running = self.runnig
        if running:
            with self.lock:
                self.runnig = False
            self.join()
        self.deinit()

    def __del__(self):
        self.stop()

def main():
    port = "6000"
    sampling_interval_sec = 1
    num_samples = 10
    power_colletor = PowerCollector(sampling_interval_sec, port)
    power_colletor.start()
    for i in range(num_samples):
        print(f"#{i}: {power_colletor.get_cur_power()}")
    
    data = power_colletor.get_all_powers()
    num = data["num_samples"]
    samples = data["sample_interval"]
    print(f"All Powers (samples: {num}, inteval_sec: {samples})")
    for i in data["powers"]:
        print(i)
    power_colletor.stop()
if __name__ == "__main__":
    main()