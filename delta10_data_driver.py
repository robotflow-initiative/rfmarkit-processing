import collections
from numpy.core.fromnumeric import shape
from numpy.ma.core import count
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Union
import logging
logging.basicConfig(level=logging.INFO)

class DataCounter:
    keys = ['IMU.IMU_ACCX', 'IMU.IMU_ACCY', 'IMU.IMU_ACCZ', 'IMU.FDI_ROLL', 'IMU.FDI_PITCH', 'IMU.FDI_YAW']

    def __init__(self, keys: List[str] = None) -> None:
        if keys is not None:
            self.keys = keys

        self.map = {key: idx for idx, key in enumerate(self.keys)}

        self.count = np.zeros(shape=(len(self.keys)), dtype=np.int64)
        self.time = np.zeros(shape=(len(self.keys)), dtype=np.float64)
        self.value = np.zeros(shape=(len(self.keys)), dtype=np.float64)

    def set(self, key, value, time) -> bool:
        if self.count[self.map[key]] == 0:
            self.count[self.map[key]] = 1
            self.value[self.map[key]] = value
            self.time[self.map[key]] = time
            return True
        else:
            self.reset()
            return False

    def reset(self):
        self.count = np.zeros(shape=(len(self.keys)), dtype=np.int64)
        self.time = np.zeros(shape=(len(self.keys)), dtype=np.float64)
        self.value = np.zeros(shape=(len(self.keys)), dtype=np.float64)

    @property
    def complete(self) -> bool:
        return np.all(self.count == 1)

    @property
    def avg_time(self) -> float:
        return np.mean(self.time)

    @property
    def data_point(self) -> Dict[str, float]:
        res = {key: self.value[self.map[key]] for key in self.keys}
        res = {
            **res,
            **{
                'timestamp': self.avg_time,
            }
        }
        return res


class Delta10DataDriver:
    KEY_MAPPING = {
        'id': None,
        'timestamp': 'timestamp',
        'accel_x': 'IMU.IMU_ACCX',
        'accel_y': 'IMU.IMU_ACCY',
        'accel_z': 'IMU.IMU_ACCZ',
        'gyro_x': None,
        'gyro_y': None,
        'gyro_z': None,
        'mag_x': None,
        'mag_y': None,
        'mag_z': None,
        'roll': 'IMU.FDI_ROLL',
        'pitch': 'IMU.FDI_PITCH',
        'yaw': 'IMU.FDI_YAW',
        'start_timestamp': None,
        'uart_buffer_len': None
    }

    def __init__(self, freq: float = 100.0) -> None:
        self.freq = freq
        self.dt = 1 / self.freq

    def __call__(self, path: str) -> Any:
        df = pd.read_csv(path)
        data_np = np.array(df)
        data_points = []
        # collections = {key: [] for key in filter(lambda x: x is not None, self.KEY_MAPPING.values())}
        counter = DataCounter()

        for line in data_np:
            if counter.complete:
                data_points.append(counter.data_point)
                counter.reset()
            t, _, entry_name, val = line
            if counter.set(entry_name, val, t / 1000) != True:
                logging.warn(f"Corruption occurred around {t / 1000}")

        length = len(data_points)
        res = {key: np.zeros(shape=(length, 1)) for key in self.KEY_MAPPING.keys()}

        for valid_key in list(filter(lambda x: self.KEY_MAPPING[x] is not None, self.KEY_MAPPING.keys())):
            for idx, point in enumerate(data_points):
                res[valid_key][idx,...] = point[self.KEY_MAPPING[valid_key]]

        res['timestamp'] = np.expand_dims(np.linspace(0, length * self.dt, length), 1).astype(np.float64)
        res['start_timestamp'] = np.copy(res['timestamp'])
        res['uart_buffer_len'] = np.zeros_like(res['timestamp']).astype(np.int64)

        filename = 'delta10_' + os.path.splitext(os.path.basename(path))[0] + '.npz'

        np.savez(os.path.join(os.path.dirname(path), filename), **res)


if __name__ == '__main__':
    driver = Delta10DataDriver(freq=200)
    driver('/Users/liyutong/projectExchange/imu-python-tools/delta10_mem/imu_log.csv')