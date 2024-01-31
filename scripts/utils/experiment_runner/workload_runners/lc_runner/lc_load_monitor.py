import zmq
import time
import threading
import json 
import copy

class LoadMonitor(threading.Thread):
    load = 0
    running = False
    sock_ = zmq.Socket
    lock = threading.Lock()
    def __init__(self, address: str = "localhost", port: str = "5952"):
        self.context = zmq.Context()
        
        self.sock_ = self.context.socket(zmq.SUB)
        self.addr = "tcp://"+address+":"+port
        try:
            self.sock_.connect(self.addr)
            self.sock_.setsockopt(zmq.SUBSCRIBE, b'server')
            self.sock_.setsockopt(zmq.CONFLATE, 1)
            self.poller = zmq.Poller()
            self.poller.register(self.sock_, zmq.POLLIN)
            
        except Exception as e:
            print(f"Could not connect to tcp://{address}:{port} error: {e}")
        threading.Thread.__init__(self)
    
    def run(self):
        print("LC server load monitor is running ...", flush=True)
        with self.lock:
            self.running = True
        
        while True:
            socks = dict(self.poller.poll(5))

            if self.sock_ in socks and socks[self.sock_] == zmq.POLLIN:
                en = self.sock_.recv_string()
                # print(en, flush=True)
                load = json.loads(self.sock_.recv_string())
                # print(load, flush=True)
                with self.lock:
                    self.load = int(load['requests'])
                    # print(self.load)
            with self.lock:
                if self.running == False:
                    break
    def stop(self):
        with self.lock:
            if self.running:
                self.running = False
        if self.is_alive():
            self.join()
        self.sock_.close()
        self.context.term()
        # self.context.destroy()
        

    def get_load(self):
        load = 0
        with self.lock:
            load = self.load
        return load

    def __del__(self):
        self.stop()

if __name__ == "__main__":
    f = LoadMonitor()
    f.start()
    ind = 0
    while ind < 10:
        print(f.get_load())
        time.sleep(1)
        ind += 1
    f.stop()

