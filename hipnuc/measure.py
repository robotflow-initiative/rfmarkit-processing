import serial
import socket
from hipnuc_module import *
import time
import argparse
import select
import logging

logging.basicConfig(level=logging.INFO)

PORT = "/dev/cu.usbserial-124440"
CONFIG = {
    "port": PORT,
    "baudrate": 921600,
    "report_datatype": {
        "imusol": True,
        "gwsol": False,
        "id": True,
        "acc": False,
        "gyr": False,
        "mag": False,
        "euler": False,
        "quat": False,
        "imusol_raw": True,
        "gwsol_raw": True
    }
}
MAX_LISTEN = 8
POLL_READ = select.POLLIN | select.POLLPRI | select.POLLHUP | select.POLLERR

def record_csv(measurement_filename: str):
    IMU: hipnuc_module = hipnuc_module(CONFIG)
    logging.info("Press Ctrl-C to terminate while statement.")
    logging.info(f'Saving measurement to {measurement_filename}')

    try:
        #create csv file
        IMU.create_csv(measurement_filename)

        with open(measurement_filename, 'a') as f:
            while True:
                try:
                    data = IMU.get_module_data(10)
                    #write to file as csv format, only work for 0x91, 0x62(IMUSOL), or there will be error
                    IMU.write2csv_handle(data, f)

                except:
                    logging.error("Error")
                    IMU.close()
                    break

    except KeyboardInterrupt:
        logging.warn("Serial is closed.")
        IMU.close()
        pass


def start_socket_service(server_addr: str):
    server_socket: socket.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if server_socket.fileno() < 0:
        logging.error('Socket error')
        return
    if os.path.exists(server_addr):
        os.unlink(server_addr)
    if server_socket.bind(server_addr):
        logging.error('Connect error')
    server_socket.listen(1)

    poller: select.poll  = select.poll()
    poller.register(server_socket.fileno(), POLL_READ)

    try:
        while True:
            epoll_list = poller.poll(1000)
            for fd, events in epoll_list:
                if events & (select.POLLIN | select.POLLPRI) and fd is server_socket.fileno():
                    client_socket, client_address = server_socket.accept()
                    while True:
                        res = client_socket.recv(1024)
                        if len(res) <= 0:
                            break
                        print(res)
                    logging.info(f"Client {client_address}closed connection")
                else:
                    print(events)
            pass
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, default='chlog.csv')
    args: argparse.Namespace = parser.parse_args()
    # record_csv(args.file)
    start_socket_service('./sock')
