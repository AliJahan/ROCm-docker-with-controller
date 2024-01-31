import zmq

class LCController:
    def __init__(self, address: str = "localhost", port: str = "5951"):
        context = zmq.Context()
        self.sock_ = context.socket(zmq.DEALER)
        try:
            self.sock_.connect("tcp://"+address+":"+port)
        except:
            print(f"Could not connect to tcp://{address}:{port}")
        self.loaded_gpus = list()

    def add_gpu(self, model, gpu):
        # if gpu in self.loaded_gpus:
        #     return True
        command = "add_worker"
        val = str(gpu)
        self.sock_.send_string(model, zmq.SNDMORE)
        self.sock_.send_string(command, zmq.SNDMORE)
        self.sock_.send_string(val)
        rep =  self.sock_.recv_string()
        if rep == "SUCCEED":
            # self.loaded_gpus.append(gpu)
            return True
        return False

    def set_batch_size(self, model, batch_size):
        command = "set_batch_size"
        val = str(batch_size)
        self.sock_.send_string(model, zmq.SNDMORE)
        self.sock_.send_string(command, zmq.SNDMORE)
        self.sock_.send_string(val)
        rep =  self.sock_.recv_string()
        if rep == "SUCCEED":
            return True
        return False
    def remove_gpu(self, model, gpu):
        # if gpu not in self.loaded_gpus:
        #     return True
        command = "remove_worker"
        val = str(gpu)
        self.sock_.send_string(model, zmq.SNDMORE)
        self.sock_.send_string(command, zmq.SNDMORE)
        self.sock_.send_string(val)
        rep =  self.sock_.recv_string()
        if rep == "SUCCEED":
            # self.loaded_gpus.remove(gpu)
            return True
        return False
    
    def pause_gpu(self, model, gpu):
        # if gpu not in self.loaded_gpus:
        #     return True
        command = "pause_worker"
        val = str(gpu)
        self.sock_.send_string(model, zmq.SNDMORE)
        self.sock_.send_string(command, zmq.SNDMORE)
        self.sock_.send_string(val)
        rep =  self.sock_.recv_string()
        if rep == "SUCCEED":
            # self.loaded_gpus.remove(gpu)
            return True
        return False
    
    def resume_gpu(self, model, gpu):
        # if gpu not in self.loaded_gpus:
        #     return True
        command = "resume_worker"
        val = str(gpu)
        self.sock_.send_string(model, zmq.SNDMORE)
        self.sock_.send_string(command, zmq.SNDMORE)
        self.sock_.send_string(val)
        rep =  self.sock_.recv_string()
        if rep == "SUCCEED":
            # self.loaded_gpus.remove(gpu)
            return True
        return False


if __name__ == "__main__":
    import sys
    model = str(sys.argv[1])
    f = LCController()
    if sys.argv[2] == "load":
        print(f.add_gpu(model, int(sys.argv[3])))
        print(f.set_batch_size(model, int(sys.argv[4])))
    elif sys.argv[2] == "unload":
        print(f.remove_gpu(model, int(sys.argv[3])))
    #print(f.set_batch_size(int(sys.argv[1])))
