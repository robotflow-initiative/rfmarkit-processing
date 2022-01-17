import socket
import multiprocessing as mp

TCP_BUFF_SZ: int = 1024
DEBUG: bool = True

def tcp_process_task(client_socket: socket.socket):

    while True:
        data = client_socket.recv(TCP_BUFF_SZ)
        if len(data) <= 0:
            print("[ Info ] Client disconnected")
            break
        if len(data) > 0:
            print(data.decode(encoding='ascii'))
            
    client_socket.close()
    return

def tcp_listen_task(address: str, port: int, max_listen: int=64) -> None:
    server_socket:socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((address, port))
    server_socket.listen(max_listen)
    print("[ Info ] Binding address {}:{}".format(address, port))
    while True:
        client_socket, (client_address, client_port) = server_socket.accept()
        print("[ Info ] New client {}:{}".format(client_address, client_port))
        client_thread = mp.Process(None, tcp_process_task, "tcp_process_{}:{}".format(client_address, client_port), (client_socket, ))
        client_thread.start()


if __name__ == '__main__':
    tcp_listen_task('0.0.0.0', 18888)