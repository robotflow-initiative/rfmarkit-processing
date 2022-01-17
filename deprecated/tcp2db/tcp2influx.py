import socket
import multiprocessing as mp
from datetime import datetime
import json
import uuid
import argparse

from influxdb_client import InfluxDBClient, Point, WritePrecision, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

from typing import Any, Union, TypeVar, Tuple, Dict, List
from influxdb_credential import ENDPOINT, TOKEN, ORG, BUCKET
from serial2json import parse_data

TCP_BUFF_SZ: int = 1024
DEBUG: bool = False


class IncompleteJsonParser(object):
    def __init__(self) -> None:
        super().__init__()
        self._buf:str = ''
    
    def __parse__(self) -> Union[None, Dict]:

        head = self._buf.find('{')
        tail: int = self._buf.find('}')

        try:
            res = json.loads(self._buf[head:tail+1])
            self._buf = self._buf[tail+1:]
            return res
        except json.JSONDecodeError as err:
            if (tail > 0):
                self._buf = self._buf[tail+1:]
            return None

    def __call__(self, json_string: str=''):
        self._buf += json_string
        res: List[Dict] = []
        while True:
            dict_obj = self.__parse__()
            if dict_obj is None:
                break
            res.append(dict_obj)
        return res



def filter_data(data: bytes, parser: IncompleteJsonParser) -> Union[dict, bytes, str]:
    points = []
    res = parser(str(data, encoding='utf-8'))
    if res is not None:
        # return [({"measurement":MEASUREMENT_BASENAME,
        #           "tags":{"id": item['id']},
        #           "time":datetime.fromtimestamp(item['timestamp'] * 1e-6).strftime("%Y-%m-%dT%H:%M:%SZ"),
        #           "fields": item}) for item in res]
        return res
    else:
        return None

def insert_data(write_api: InfluxDBClient, data: List[Dict], measurement_name: str):
    if data is not None:
        if DEBUG:
            print("[ Data ] {}".format(str(data)))
    elif data is None:
        return 
    for item in data:
        point = Point(measurement_name)
        point.tag("id", item['id'])
        point.time(datetime.fromtimestamp(item['timestamp'] * 1e-6), WritePrecision.NS)
        point.field("accel_x", 1.2)
        write_api.write(BUCKET, ORG, point)
    

def tcp_process_task(client_socket: socket.socket, measurement_name: str):
    client = InfluxDBClient(url=ENDPOINT, token=TOKEN)
    parser = IncompleteJsonParser()
    write_api = client.write_api(write_options=SYNCHRONOUS)
    while True:
        data = client_socket.recv(TCP_BUFF_SZ)
        if len(data) <= 0:
            print("[ Info ] Client disconnected")
            break
        if len(data) > 0:
            if DEBUG:
                print(data.decode(encoding='ascii'))
            else:
                insert_data(write_api, filter_data(data, parser), measurement_name)

    client_socket.close()
    return

def tcp_listen_task(address: str, port: int, measurement_name: str, max_listen: int=64) -> None:
    server_socket:socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((address, port))
    server_socket.listen(max_listen)
    print("[ Info ] Binding address {}:{}".format(address, port))
    while True:
        client_socket, (client_address, client_port) = server_socket.accept()
        print("[ Info ] New client {}:{}".format(client_address, client_port))
        client_thread = mp.Process(None, tcp_process_task, "tcp_process_{}:{}".format(client_address, client_port), (client_socket, measurement_name, ))
        client_thread.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='')
    args = parser.parse_args()
    if args.name == '':
        MEASUREMENT_BASENAME = 'imu_mem_'+ datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    else:
        MEASUREMENT_BASENAME = args.name

    tcp_listen_task('0.0.0.0', 18888, MEASUREMENT_BASENAME)
