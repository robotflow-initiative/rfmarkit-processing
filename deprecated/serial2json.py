from serial import Serial
import numpy as np
import os
import time
import json
import datetime
from typing import List
import multiprocessing as mp
import tqdm

DATA_FORMAT = {
    "accel_x": [5, 4],
    "accel_y": [7, 6],
    "accel_z": [9, 8],
    "gyro_x": [11, 10],
    "gyro_y": [13, 12],
    "gyro_z": [15, 14],
    "roll": [17, 16],
    "pitch": [19, 18],
    "yaw": [21, 20],
    "temp": [24, 23],
    "mag_x": [26, 25],
    "mag_y": [28, 27],
    "mag_z": [30, 29],
}
DATA_DIVIDER = {
    "accel_x": 4 / 65536,
    "accel_y": 4 / 65536,
    "accel_z": 4 / 65536,
    "gyro_x": 500 / 65536,
    "gyro_y": 500 / 65536,
    "gyro_z": 500 / 65536,
    "roll": 1 / 100,
    "pitch": 1 / 100,
    "yaw": 1 / 100,
    "temp": 1 / 100,
    "mag_x": 4 / 65536,
    "mag_y": 4 / 65536,
    "mag_z": 4 / 65536,
}


def parse_data(data):
    res: dict = {
        "accel_x": 0.0,
        "accel_y": 0.0,
        "accel_z": 0.0,
        "gyro_x": 0.0,
        "gyro_y": 0.0,
        "gyro_z": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "temp": 0.0,
        "mag_x": 0.0,
        "mag_y": 0.0,
        "mag_z": 0.0,
    }
    for key in res.keys():
        value = data[DATA_FORMAT[key][0]] * 256 + data[DATA_FORMAT[key][1]]
        value = -(65536 - value) if value >= 32768 else value
        res[key] = value * DATA_DIVIDER[key]
    return res


def append_chksum(ary: bytearray) -> bytearray:
    ary.append(sum(ary) % 256)
    return ary


class GY95:
    ADDR = 0xa4
    READ_OP = 0x03
    REG_THRESH = 0x2c
    DEFAULT_START_REG = 0x14

    def __init__(self, port: str, baud: int = 115200) -> None:
        self.port = port
        self.buf: np.ndarray = np.zeros(shape=(40), dtype=np.uint8)
        self.cursor: int = 0
        self.ser: Serial = Serial(port, baud, timeout=1)
        self.start_reg: int = 0
        self.length: int = 0
        self.sign: bool = False

    def _reset(self):
        self.cursor = 0
        self.length = 0
        self.start_reg = 0
        self.sign = False
        self.buf.fill(0)

    def calibrate(self):
        if self.ser.writable():
            self.ser.write(append_chksum(bytearray([0xa4, 0x06, 0x02, 0x02])))  # Rate 100Hz
            time.sleep(0.1)
            self.ser.write(append_chksum(bytearray([0xa4, 0x06, 0x03, 0x00])))  # Auto update
            time.sleep(0.1)
            self.ser.write(append_chksum(bytearray([0xa4, 0x06, 0x07, 0x80])))  # Mount horizontally
            time.sleep(0.1)
            self.ser.write(append_chksum(bytearray([0xa4, 0x06, 0x05, 0x57])))  # Calibrate
            time.sleep(7)

    def read(self) -> np.ndarray:
        self._reset()
        while self.ser.readable():
            self.buf[self.cursor] = ord(self.ser.read())
            if self.cursor == 0:
                if self.buf[0] != self.ADDR:
                    self._reset()
                    continue
            elif self.cursor == 1:
                if self.buf[1] != self.READ_OP:
                    self._reset()
                    continue

            elif self.cursor == 2:
                if self.buf[2] < self.REG_THRESH:
                    self.start_reg = self.buf[2]
                else:
                    self._reset()
                    continue

            elif self.cursor == 3:
                if self.start_reg + self.buf[3] < self.REG_THRESH:
                    self.length = self.buf[3]
                else:
                    self._reset()
                    continue

            else:
                if self.length + 5 == self.cursor:
                    self.sign = True

            if self.sign:
                self.sign = False
                if sum(self.buf[:self.cursor - 1]) % 0x100 == self.buf[self.cursor - 1]:
                    return self.buf
                self.cursor = 0
            else:
                self.cursor += 1


def serial_process_task(port: str, filename: str):
    gy = GY95(port)
    gy.calibrate()
    print(port, filename)

    record = []

    filename += "_IMU_record.json"
    start_time = time.time()

    with tqdm.tqdm(range(1)) as pbar:
        try:
            while True:
                raw_data = gy.read()
                parsed_data = parse_data(raw_data)
                pbar.set_description("accel-x: {:.2f}, accel-y: {:.2f}, accel-z: {:.2f}".format(parsed_data['accel_x'], parsed_data['accel_y'],
                                                                     parsed_data['accel_z']))

                parsed_data["time"] = time.time() - start_time
                record.append(parsed_data)
        except KeyboardInterrupt:
            with open(filename, 'w') as f:
                json.dump(record, f)


def entry(stm32_ports: List[str], filename: str) -> None:
    try:
        for port in stm32_ports:
            filename = filename + "_" + os.path.basename(port)
            client_thread = mp.Process(None, serial_process_task, "serial_process_task_{}".format(port), (
                port,
                filename,
            ))
            client_thread.start()
    except KeyboardInterrupt:
        print('Ctrl-C pressed')


if __name__ == '__main__':
    # GY95_ADDR = chr(0xa4)
    # ser = Serial('/dev/cu.usbserial-13110', 115200, timeout=1)
    # while(True):
    #     data = ser.read_until(GY95_ADDR)
    #     print(len(data))
    STM32_PORTS: list = ['/dev/cu.usbserial-124440']
    filename = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    entry(STM32_PORTS, filename)
