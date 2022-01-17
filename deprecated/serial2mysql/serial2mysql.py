import multiprocessing as mp
import pymysql
from serial import Serial
from typing import Any, Union, TypeVar, Tuple, List
import json
from mysql_credential import MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD, MYSQL_BASE_STRING

MYSQL_BASE_STRING = "INSERT INTO test(data) VALUES ('{}')"

def filter_data(data: bytes) -> Union[dict, bytes, str, None]:
    try:
        imu_data_dict = json.loads(data)
        return imu_data_dict
    except:
        if len(data) > 0:
            print(data)
            return None

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

def serial_process_task(ser: Serial):
    conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, db=MYSQL_DB, user=MYSQL_USER, password=MYSQL_PASSWORD)
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    while True:
        data = ser.readline()
        if len(data) <= 0:
            print("[ Info ] Client disconnected")
            break
        if len(data) > 0:
            data = filter_data(data)
            if len(data) > 0:
                insert_data(conn, cursor, data)

    cursor.close()            
    conn.close()
    return


def entry(stm32_ports: List[str]) -> None:
    for port in stm32_ports:
        try:
            ser = Serial(port, 115200, timeout=1)
        except Exception as err:
            print(err)
            continue
        client_thread = mp.Process(None, serial_process_task, "serial_process_task_{}".format(port), (ser, ))
        client_thread.start()


if __name__ == '__main__':
    STM32_PORTS: list = ['/dev/cu.usbserial-120']
    entry(STM32_PORTS)