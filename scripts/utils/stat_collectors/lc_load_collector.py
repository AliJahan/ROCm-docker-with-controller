import zmq
import json
import threading
import multiprocessing

class LCLoadCollector(threading.Thread):
    lock = threading.Lock()
    queue = multiprocessing.Queue()
    lc_stat_socket = None
    poller = None
    def __init__(
            self,
            load_broadcaster_ip: str,
            load_broadcaster_port: str
        ):
        self.load_broadcaster_port = load_broadcaster_port
        self.load_broadcaster_ip = load_broadcaster_ip
        self.initialized = self.init()
        self.runnig = False
        self.num_samples = 0
        self.powers_list = list()
        threading.Thread.__init__(self)
    def is_initted(self):
        return self.initialized
    
    def init(self):
        self.lc_stat_socket, self.poller = self.setup_socket()
        if  self.lc_stat_socket is None:
            return False
    def deinit(self):
        if self.lc_stat_socket is not None:
            self.lc_stat_socket.close()
            del self.lc_stat_socket
        self.ctx.term()
        

    def setup_socket(self):
        print(f"Connecting to power broadcaster {self.load_broadcaster_ip}:{self.load_broadcaster_port} ... ", end="")
        self.ctx = zmq.Context.instance()
        publisher = self.ctx.socket(zmq.SUB)
        poller = None
        try:
            publisher.connect(f"tcp://{self.load_broadcaster_ip}:{self.load_broadcaster_port}")
            publisher.setsockopt(zmq.SUBSCRIBE, b"server")
            # publisher.setsockopt(zmq.CONFLATE, 1)
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

            if self.lc_stat_socket in socks and socks[self.lc_stat_socket] == zmq.POLLIN:
                msg = None
                try:
                    msg = self.lc_stat_socket.recv_string() # channel name msg will be ignored
                    msg = self.lc_stat_socket.recv_string()
                except zmq.ZMQError as e:
                    if e.errno == zmq.ETERM:
                        print("ZMQ socket interrupted/terminated, Quitting...")
                    else:
                        print(f"ZMQ socket error: {e}, Quitting...")
                    break
                # It should not be None here (just in case)
                if msg is None:
                    continue
                # Record current load(rps) data
                rcvd_power_data = json.loads(msg)
                with self.lock:
                    if self.queue.empty() is False:
                        self.queue.get()
                self.queue.put(int(rcvd_power_data['requests']))
            with self.lock:
                if self.runnig == False:
                    break
            # time.sleep(self.collection_interval_sec)

    def get_cur_load(self): # gets the last power published
        return self.queue.get()

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
    ip = "172.20.0.9"
    port = "5952"
    num_samples = 10
    power_colletor = LCLoadCollector(
        load_broadcaster_ip=ip,
        load_broadcaster_port=port
    )
    power_colletor.start()
    for i in range(num_samples):
        print(f"#{i}: {power_colletor.get_cur_load()}")
    power_colletor.stop()
if __name__ == "__main__":
    main()