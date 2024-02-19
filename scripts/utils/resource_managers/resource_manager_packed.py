import zmq
import time
from enum import Enum

class GPUWorkloadType(Enum):
    FREE = 0
    BE = 1
    LC = 2

class GPUResourceManagerPacked:
    SLEEP_AFTER_SEND_MSG_SEC = 1
    resrouce_controller_socket = None
    remote_header = "remote_docker_runner"
    gpus_masks = dict()
    max_num_gpus = 8
    num_shader_engins = 4
    max_cus = 60
    apps_freq_changed = dict()
    def __init__(
            self,
            remote_control_ip: str = "172.20.0.6",
            remote_control_port: str = "5000"
        ):
        self.remote_control_ip = remote_control_ip
        self.remote_control_port = remote_control_port

        # Set masks for all GPU masks (MI50 has 60 CUs)
        for i in range(self.max_num_gpus): 
            self.gpus_masks[i] = {GPUWorkloadType.BE: dict(), GPUWorkloadType.LC: dict(), GPUWorkloadType.FREE: dict()}
        for i in range(self.max_num_gpus): 
            for se in range(self.num_shader_engins): # MI50 has 4 shader engines (se)
                self.gpus_masks[i][GPUWorkloadType.FREE][se] = [(se+1)+self.num_shader_engins*k for k in range(15)] # MI50 has 15 CUs per shader
                self.gpus_masks[i][GPUWorkloadType.BE][se] = list()
                self.gpus_masks[i][GPUWorkloadType.LC][se] = list()

        self.resrouce_controller_socket = self.setup_socket()

    def convert_bin2hex_str(self, bin_str):
        _hex = str(hex(int(bin_str, 2)))[2:]
        _len = len(_hex)
        for i in range(_len, 8): # 32bits == 8bytes
            _hex = '0'+_hex
        return _hex
    def generate_bin_str(self, cu_list):
        cu_list_all = list()
        for se in range(self.num_shader_engins):
            cu_list_all += cu_list[se]

        mask = ""
        for i in range(64,0,-1):
            if i in cu_list_all:
                mask += '1'
            else:
                mask += '0'
        return mask

    def get_current_cus(self, gpu: int, is_be = False):
        cur_cus = 0

        if is_be:
            for se in range(self.num_shader_engins):
                cur_cus += len(self.gpus_masks[gpu][GPUWorkloadType.BE][se])
        else:
            for se in range(self.num_shader_engins):
                cur_cus += len(self.gpus_masks[gpu][GPUWorkloadType.LC][se])

        return cur_cus

    def add_cu(self, app_name: str, gpu: int, cus: int, is_be = False):
        if cus < 0:
            return False
        
        if gpu not in self.gpus_masks:
            print(f"\t- [GPUResourceManager]: ERROR requested gpu ({gpu}) is not available!")
            return False

        total_avail_cus = 0
        for se in range(self.num_shader_engins):
            total_avail_cus += len(self.gpus_masks[gpu][GPUWorkloadType.FREE][se])

        if total_avail_cus < cus:
            print(f"\t- [GPUResourceManager]: ERROR requested cus ({cus}) is less than the available cus({total_avail_cus})")
            return False

        mask = ""
        if is_be:
            cus_needed = cus
            while cus_needed:
                for se in range(self.num_shader_engins-1, -1, -1): # for BE we pick cus from higher indexed SEs
                    if len(self.gpus_masks[gpu][GPUWorkloadType.FREE][se]) > 0:
                        self.gpus_masks[gpu][GPUWorkloadType.FREE][se] = sorted(self.gpus_masks[gpu][GPUWorkloadType.FREE][se])
                        self.gpus_masks[gpu][GPUWorkloadType.BE][se].append(self.gpus_masks[gpu][GPUWorkloadType.FREE][se][-1])
                        self.gpus_masks[gpu][GPUWorkloadType.FREE][se].pop()
                        cus_needed -= 1
                        break
            mask = self.generate_bin_str(self.gpus_masks[gpu][GPUWorkloadType.BE])
        else:
            cus_needed = cus
            while cus_needed:
                for se in range(self.num_shader_engins): # for LC we pick cus from lower indexed SEs
                    if len(self.gpus_masks[gpu][GPUWorkloadType.FREE][se]) > 0:
                        self.gpus_masks[gpu][GPUWorkloadType.FREE][se] = sorted(self.gpus_masks[gpu][GPUWorkloadType.FREE][se])
                        self.gpus_masks[gpu][GPUWorkloadType.LC][se].append(self.gpus_masks[gpu][GPUWorkloadType.FREE][se][0])
                        self.gpus_masks[gpu][GPUWorkloadType.FREE][se].pop(0)
                        cus_needed -= 1
                        break
            mask = self.generate_bin_str(self.gpus_masks[gpu][GPUWorkloadType.LC])

        assert len(mask) == 64, f"generated mask: {mask} is longer than 64 bits!"
        mask1 = self.convert_bin2hex_str(mask[:32])
        mask0 = self.convert_bin2hex_str(mask[32:])

        print(f"\t- [GPUResourceManager]: Mask for app:{app_name} generated. (mask0={mask0} mask1={mask1})", flush=True)

        return self.set_mask(
            app_name=app_name,
            gpu=gpu,
            cumask_full_hex0=mask0,
            cumask_full_hex1=mask1
        )

    def remove_cu(self, app_name: str, gpu: int, cus: int, is_be = False):
        if cus < 0:
            print(f"\t- [GPUResourceManager]: ERROR requested cus ({cus}) is less than zero")
            return False

        if cus == 0:
            return True

        if gpu not in self.gpus_masks:
            print(f"\t- [GPUResourceManager]: ERROR requested gpu ({gpu}) is not available!")
            return False
        
        total_owned_cus = 0
        if is_be:
            for se in range(self.num_shader_engins):
                total_owned_cus += len(self.gpus_masks[gpu][GPUWorkloadType.BE][se])
        else:
            for se in range(self.num_shader_engins):
                total_owned_cus += len(self.gpus_masks[gpu][GPUWorkloadType.LC][se])

        if total_owned_cus < cus:
            print(f"\t- [GPUResourceManager]: ERROR requested cus ({cus}) is less than the available cus to remove ({total_owned_cus})")
            return False
        mask = ""
        if is_be:
            cus_released = cus
            while cus_released:
                for se in range(self.num_shader_engins): # for BE we release cus from lower indexed SEs
                    if len(self.gpus_masks[gpu][GPUWorkloadType.BE][se]) > 0:
                        self.gpus_masks[gpu][GPUWorkloadType.BE][se] = sorted(self.gpus_masks[gpu][GPUWorkloadType.BE][se])
                        self.gpus_masks[gpu][GPUWorkloadType.FREE][se].append(self.gpus_masks[gpu][GPUWorkloadType.BE][se][0])
                        self.gpus_masks[gpu][GPUWorkloadType.BE][se].pop(0)
                        cus_released -= 1
                        break
            mask = self.generate_bin_str(self.gpus_masks[gpu][GPUWorkloadType.BE])
        else:
            cus_released = cus
            while cus_released:
                for se in range(self.num_shader_engins-1, -1, -1): # for LC we pick cus from lower indexed SEs
                    if len(self.gpus_masks[gpu][GPUWorkloadType.LC][se]) > 0:
                        self.gpus_masks[gpu][GPUWorkloadType.LC][se] = sorted(self.gpus_masks[gpu][GPUWorkloadType.LC][se])
                        self.gpus_masks[gpu][GPUWorkloadType.FREE][se].append(self.gpus_masks[gpu][GPUWorkloadType.LC][se][-1])
                        self.gpus_masks[gpu][GPUWorkloadType.LC][se].pop()
                        cus_released -= 1
                        break
            
            mask = self.generate_bin_str(self.gpus_masks[gpu][GPUWorkloadType.LC])

        assert len(mask) == 64, f"generated mask: {mask} is longer than 64 bits!"
        mask1 = self.convert_bin2hex_str(mask[:32])
        mask0 = self.convert_bin2hex_str(mask[32:])

        print(f"\t- [GPUResourceManager]: Mask for app:{app_name} generated. (mask0={mask0} mask1={mask1})", flush=True)
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
        if gpu not in self.gpus_masks:
            print(f"Gpu ({gpu}) does not exists", flush=True)
            return False

        if freq < 5 or freq > 225:
            print(f"Freq provided ({freq}) is not withing range (5,225)", flush=True)
            return False

        # Warning check:
        if len(self.gpus_masks[gpu][GPUWorkloadType.BE]) > 0 and len(self.gpus_masks[gpu][GPUWorkloadType.LC]) > 0:
            print(f"WARNNING: Both BE and LC apps are located in this GPU, setting gpu {gpu} freq to {freq} may impact both apps performance", flush=True)
        # for cleanup
        if app_name not in self.apps_freq_changed:
            self.apps_freq_changed[app_name] = list()
            if gpu not in self.apps_freq_changed[app_name]:
                self.apps_freq_changed[app_name].append(gpu)
        return self.send_msg(app_name, f"SET_FREQ:{gpu}:{freq}")
    
    def cleanup(self):
        for app_name in self.apps_freq_changed: # app_name unnecessary since freq is set per gpu
            for gpu in self.apps_freq_changed[app_name]:
                self.set_freq(app_name=app_name, gpu=gpu, freq=225)
def test_resrouce_manager_packet():
    mgr = GPUResourceManagerPacked()
    # return
    print("Adding BE")
    # time.sleep(10)
    for i in range(3):
        print(f"{i}--- add 5 to BE")
        mgr.add_cu("asd", 0, 60, True)
        input()
        print(f"{i}--- remove 2 from BE")
        mgr.remove_cu("asd", 0, 48, True)
        input()
        print(f"{i}--- add 4 to LC")
        mgr.add_cu("asd", 0, 50, False)
        print(f"{i}--- remove 3 from LC")
        mgr.remove_cu("asd", 0, 48, False)
        print(f"@@@@---@@@")
    
if __name__ == "__main__":
    # test_resrouce_manager()
    test_resrouce_manager_packet()