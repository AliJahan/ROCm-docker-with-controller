import zmq
import sys
import time
import json

def setup_sockets(port: str):
    print(f"Creating socket on port: {port}")
    ctx = zmq.Context.instance()
    publisher = ctx.socket(zmq.SUB)
    
    print(f"Binding ...", end="")
    try:
        publisher.connect(f"tcp://localhost:{port}")
        publisher.setsockopt(zmq.SUBSCRIBE, b"")
        print(" Success!")
        return publisher
    except Exception as e:
        print(f" Failed! error: {e}")
    
    return None
def broadcast_power(publish_socket):
    
    print("Subscribing the power...")
    while True:
        try:
            msg = publish_socket.recv_string()
            power_data = json.loads(msg)
            print(power_data)
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                print("ZMQ socket interrupted/terminated, Quitting...")
            else:
                print(f"ZMQ socket error: {e}, Quitting...")
            break
    
def main():
    port = "6000"
    if len(sys.argv) == 2:
        port = str(sys.argv[1])
    elif len(sys.argv) > 2:
        print("usage: python3 power_broadcaster_test [port(default=6000)]")
        return -1
    
    publish_socket = setup_sockets(port)
    if publish_socket is None:
        return -1
    broadcast_power(publish_socket)

if __name__ == "__main__":
    main()