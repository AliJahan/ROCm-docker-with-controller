import zmq
import time
from enum import Enum

class WorkloadType(Enum):
    FREE = 0
    BE = 1
    LC = 2

class ResourceManager:
    SLEEP_AFTER_SEND_MSG_SEC = 1
    resrouce_controller_socket = None
    remote_header = "remote_docker_runner"
    gpus_masks = dict()
    max_num_gpus = 8
    max_cus = 60
    apps_freq_changed = dict()
    def __init__(
            self,
            remote_control_ip: str = "172.20.0.6",
            remote_control_port: str = "5000"
        ):
        self.remote_control_ip = remote_control_ip
        self.remote_control_port = remote_control_port
        self.resrouce_controller_socket = self.setup_socket()
        # Set masks for all GPU masks (MI50 has 60 CUs)
        for i in range(self.max_num_gpus): 
            self.gpus_masks[i] = {WorkloadType.BE: list(), WorkloadType.LC: list(), WorkloadType.FREE: [i for i in range(1, self.max_cus+1)]}

    def convert_bin2hex_str(self, bin_str):
        _hex = str(hex(int(bin_str, 2)))[2:]
        _len = len(_hex)
        for i in range(_len, 8): # 32bits == 8bytes
            _hex = '0'+_hex
        return _hex
    def generate_bin_str(self, cu_list):
        mask = ""
        for i in range(64,0,-1):
            if i in cu_list:
                mask += '1'
            else:
                mask += '0'
        return mask

    def add_cu(self, app_name: str, gpu: int, cus: int, is_be = False):
        if gpu not in self.gpus_masks:
            return False
        if len(self.gpus_masks[gpu][WorkloadType.FREE]) < cus:
            return False
        
        self.gpus_masks[gpu][WorkloadType.FREE] = sorted(self.gpus_masks[gpu][WorkloadType.FREE])

        mask = ""
        if is_be:
            self.gpus_masks[gpu][WorkloadType.BE] += self.gpus_masks[gpu][WorkloadType.FREE][-cus:]
            self.gpus_masks[gpu][WorkloadType.FREE] = self.gpus_masks[gpu][WorkloadType.FREE][:-cus]
            mask = self.generate_bin_str(self.gpus_masks[gpu][WorkloadType.BE])
        else:
            self.gpus_masks[gpu][WorkloadType.LC] += self.gpus_masks[gpu][WorkloadType.FREE][:cus]
            self.gpus_masks[gpu][WorkloadType.FREE] = self.gpus_masks[gpu][WorkloadType.FREE][cus:]
            mask = self.generate_bin_str(self.gpus_masks[gpu][WorkloadType.LC])
        assert len(mask) == 64, f"generated mask: {mask} is longer than 64 bits!"
        mask1 = self.convert_bin2hex_str(mask[:32])
        mask0 = self.convert_bin2hex_str(mask[32:])

        print(f"Mask for app:{app_name} generated. (mask0={mask0} mask1={mask1})", flush=True)

        return self.set_mask(
            app_name=app_name,
            gpu=gpu,
            cumask_full_hex0=mask0,
            cumask_full_hex1=mask1
        )
    def remove_cu(self, app_name: str, gpu: int, cus: int, is_be = False):
        if cus == 0:
            return True

        if gpu not in self.gpus_masks:
            return False

        if is_be and len(self.gpus_masks[gpu][WorkloadType.BE]) < cus:
            return False

        if not is_be and len(self.gpus_masks[gpu][WorkloadType.LC]) < cus:
            return False

        mask = ""
        if is_be:
            self.gpus_masks[gpu][WorkloadType.BE] = sorted(self.gpus_masks[gpu][WorkloadType.BE])
            self.gpus_masks[gpu][WorkloadType.FREE] += self.gpus_masks[gpu][WorkloadType.BE][:cus]
            self.gpus_masks[gpu][WorkloadType.BE] = self.gpus_masks[gpu][WorkloadType.BE][cus:]
            mask = self.generate_bin_str(self.gpus_masks[gpu][WorkloadType.BE])
        else:
            self.gpus_masks[gpu][WorkloadType.LC] = sorted(self.gpus_masks[gpu][WorkloadType.LC])
            self.gpus_masks[gpu][WorkloadType.FREE] += self.gpus_masks[gpu][WorkloadType.LC][-cus:]
            self.gpus_masks[gpu][WorkloadType.LC] = self.gpus_masks[gpu][WorkloadType.LC][:-cus]
            mask = self.generate_bin_str(self.gpus_masks[gpu][WorkloadType.LC])

        assert len(mask) == 64, f"generated mask: {mask} is longer than 64 bits!"
        mask1 = self.convert_bin2hex_str(mask[:32])
        mask0 = self.convert_bin2hex_str(mask[32:])

        print(f"Mask for app:{app_name} generated. (mask0={mask0} mask1={mask1})", flush=True)
        return self.set_mask(
            app_name=app_name,
            gpu=gpu,
            cumask_full_hex0=mask0,
            cumask_full_hex1=mask1
        )
    
    def setup_socket(self):
        self.ctx = zmq.Context.instance()
        publisher = None
        # poller = None
        print(f"Binding to {self.remote_control_ip}:{self.remote_control_port}... ", end="")
        try:
            publisher = self.ctx.socket(zmq.PUB)
            publisher.bind(f"tcp://*:{self.remote_control_port}")
            print("Success!")
            return publisher
        except Exception as e:
            print(f"Failed! error: {e}")
        return None

    def send_msg(self, channel, msg):
        print(f"\t- Sending message: ({channel}) {msg} ... ", end="", flush=True)
        rep = True
        try:
            self.resrouce_controller_socket.send_string(f"{channel}", flags=zmq.SNDMORE)
            self.resrouce_controller_socket.send_string(f"{msg}")
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                print("FAILED! error: ZMQ socket interrupted/terminated", flush=True)
            else:
                print(f"FAILED! error: ZMQ socket error: {e}", flush=True)
            rep = False
        if rep == True:
            time.sleep(self.SLEEP_AFTER_SEND_MSG_SEC)
            print(f"Done!") 
        return rep
    
    def set_mask(self, app_name, gpu, cumask_full_hex0, cumask_full_hex1):
        return self.send_msg(app_name, f"SET_CUMASK:{gpu}:{cumask_full_hex0}:{cumask_full_hex1}")
    
    def set_freq(self, app_name: str, gpu: int, freq: int):
        if freq < 5 or freq > 225:
            print(f"Freq provided ({freq}) is not withing range (5,225)", flush=True)
            return False
        # Warning check:
        if len(self.gpus_masks[gpu][WorkloadType.BE]) > 0 and len(self.gpus_masks[gpu][WorkloadType.LC]) > 0:
            print(f"WARNNING: Both BE and LC apps are located in this GPU, setting gpu {gpu} freq to {freq} may impact both apps performance", flush=True)
        # for cleanup
        if app_name not in self.apps_freq_changed:
            self.apps_freq_changed[app_name] = list()
            if gpu not in self.apps_freq_changed[app_name]:
                self.apps_freq_changed[app_name].append(gpu)
        return self.send_msg(app_name, f"SET_FREQ:{gpu}:{freq}")
    
    def cleanup(self):
        for app_name in self.apps_freq_changed:
            for gpu in self.apps_freq_changed[app_name]:
                self.set_freq(app_name=app_name, gpu=gpu, freq=225)

def test_resrouce_manager():
    mgr = ResourceManager()
    print("Adding BE")
    # time.sleep(10)
    for i in range(6):
        print(f"{i}--- add 5 to BE")
        mgr.add_cu("asd", 0, 5, True)
        print(f"{i}--- remove 2 from BE")
        mgr.remove_cu("asd", 0, 2, True)
        print(f"{i}--- add 4 to LC")
        mgr.add_cu("asd", 0, 4, False)
        print(f"{i}--- remove 3 from LC")
        mgr.remove_cu("asd", 0, 3, False)
        print(f"@@@@---@@@")
    
    # print(f"REMOVING---")
    
    # for i in range(6):
        
    #     print(f"{i}---")
    return

    print("Adding LC")
    for i in range(9):
        mgr.add_cu("asd", 0, 5, False)
        print(f"{i}---")
    mgr.add_cu("asd", 0, 4, False)
    print(mgr.add_cu("asd", 0, 1, True))
    
    # mgr.set_freq("Inference-Server", 0, 225)
    # time.sleep(1000)

if __name__ == "__main__":
    test_resrouce_manager()