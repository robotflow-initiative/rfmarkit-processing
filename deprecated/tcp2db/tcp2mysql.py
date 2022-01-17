import socket
import multiprocessing as mp
import pymysql
from typing import Any, Union, TypeVar, Tuple
from mysql_credential import MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD, MYSQL_BASE_STRING

TCP_BUFF_SZ: int = 1024
DEBUG: bool = True

# MYSQL_HOST = '127.0.0.1'
# MYSQL_PORT = 3306
# MYSQL_DB = 'IMU_DATA'
# MYSQL_USER = 'root'
# MYSQL_PASSWORD = ''


def filter_data(data: bytes) -> Union[dict, bytes, str]:
    return data

def insert_data(conn: pymysql.Connection, cursor: TypeVar, data: Tuple[Any]):
    print("[ Data ] {}".format(str(data)))
    try:
        sql = MYSQL_BASE_STRING.format(str(data, encoding='UTF-8'))
        cursor.execute(sql)
        conn.commit()
    except Exception as exce:
        print("[ Info ] Got an error when executing: {}".format(sql))
        print("|------- BEGIN EXECEPTION -------|")
        print(exce)
        print("|-------- END EXECEPTION --------|")
        conn.rollback()

def tcp_process_task(client_socket: socket.socket):
    conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, db=MYSQL_DB, user=MYSQL_USER, password=MYSQL_PASSWORD)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    while True:
        data = client_socket.recv(TCP_BUFF_SZ)
        if len(data) <= 0:
            print("[ Info ] Client disconnected")
            break
        if len(data) > 0:
            if DEBUG:
                print(data.decode(encoding='ascii'))
            else:
                insert_data(conn, cursor, filter_data(data))

    cursor.close()            
    conn.close()
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
    tcp_listen_task('0.0.0.0', 8888)