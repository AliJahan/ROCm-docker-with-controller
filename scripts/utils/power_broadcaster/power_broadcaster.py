import zmq
import sys
import time
import json
from power_monitors.cpu_power_monitor import CPUPowerMonitor
from power_monitors.gpu_power_monitor import GPUPowerMonitor

DEBUG = False

def setup_sockets(port: str):
    print(f"Creating socket on port: {port}")
    ctx = zmq.Context.instance()
    publisher = ctx.socket(zmq.PUB)
    print(f"Binding ...", end="")
    try:
        publisher.bind(f"tcp://*:{port}")
        print(" Success!")
        return publisher
    except Exception as e:
        print(f" Failed! error: {e}")
    
    return None
    

def broadcast_power(publish_socket, cpu_read_interval_msec, gpu_read_interval_msec, broad_cast_interval_sec):
    cpu_mon = CPUPowerMonitor(cpu_read_interval_msec)
    gpu_mon = GPUPowerMonitor(gpu_read_interval_msec)
    gpu_mon.start()
    cpu_mon.start()
    print("Broadcasting the power...")
    while True:
        gpu_power = gpu_mon.get_avg_power()
        cpu_power = cpu_mon.get_avg_power()
        server_power = cpu_power + gpu_power['total']
        message = "{ \"cpu\": " + str(cpu_power) +", \"gpu\": "+ json.dumps(gpu_power) +", \"total\": " + str(server_power) + "}"
        if DEBUG is True:
            print(message)
        try:
            publish_socket.send(message.encode('utf-8'))
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                print("ZMQ socket interrupted/terminated, Quitting...")
            else:
                print(f"ZMQ socket error: {e}, Quitting...")
            break
        time.sleep(broad_cast_interval_sec)
    cpu_mon.stop()
    gpu_mon.stop()

def main():
    cpu_read_interval_msec = 500
    gpu_read_interval_msec = 500
    broad_cast_interval_sec = 1
    port = "6000"
    if len(sys.argv) == 2:
        port = str(sys.argv[1])
    elif len(sys.argv) == 4:
        cpu_read_interval_msec = int(sys.argv[2])
        gpu_read_interval_msec = int(sys.argv[3])
    elif len(sys.argv) > 4:
        print("usage: python3 power_broadcaster [port(default=6000)] [cpu_power_sampling_interval_msec(default=500) gpu_power_sampling_interval_msec(default=500)]")
        return -1
    
    publish_socket = setup_sockets(port)
    if publish_socket is None:
        return -1
    broadcast_power(publish_socket, cpu_read_interval_msec, gpu_read_interval_msec, broad_cast_interval_sec)

if __name__ == "__main__":
    main()