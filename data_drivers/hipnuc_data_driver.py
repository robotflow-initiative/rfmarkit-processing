import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Union

class HIPNUCDataDriver:
    KEY_MAPPING = {'id': None,
                   'timestamp': None,
                   'accel_x':'AccX[G]',
                   'accel_y':'AccY[G]',
                   'accel_z':'AccZ[G]',
                   'gyro_x':'GyrX[deg/s]',
                   'gyro_y':'GyrY[deg/s]',
                   'gyro_z':'GyrZ[deg/s]',
                   'mag_x':'MagX[uT]',
                   'mag_y':'MagY[uT]',
                   'mag_z':'MagZ[uT]',
                   'roll':'Roll[deg]',
                   'pitch':'Pitch[deg]',
                   'yaw':'Yaw[deg]',
                   'start_timestamp': None,
                   'uart_buffer_len': None}
    def __init__(self, freq: float=200.0) -> None:
        self.freq = freq
        self.dt = 1 / self.freq

    def __call__(self, path: str) -> Any:
        df = pd.read_csv(path)
        
        res = {key:None for key in self.KEY_MAPPING.keys()}
        for key, value in self.KEY_MAPPING.items():
            if value is not None:
                res[key] = np.expand_dims(np.array(df[value], dtype=np.float64), 1)
        
        res['accel_x'] = -res['accel_x']
        res['accel_y'] = -res['accel_y']
        res['accel_z'] = -res['accel_z']

        length = len(df)
        res['timestamp'] = np.expand_dims(np.linspace(0, length * self.dt, length), 1).astype(np.float64)
        res['start_timestamp'] = np.expand_dims(np.linspace(0, length * self.dt, length), 1).astype(np.float64)
        res['uart_buffer_len'] = np.zeros_like(res['timestamp']).astype(np.int64)

        filename = 'hipnuc_' + os.path.splitext(os.path.basename(path))[0] + '.npz'

        np.savez(os.path.join(os.path.dirname(path), filename), **res)

if __name__ == '__main__':
    import glob
    driver = HIPNUCDataDriver(freq=200)
    for file in glob.glob('/Users/liyutong/projectExchange/imu-python-tools/hipnuc_mem/*.csv'):
        print(file)
        driver(file)