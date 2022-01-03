from threading import local
from typing import List, Tuple
from torch.utils.data import Dataset
import re
from typing import Dict
import pandas as pd
import os
import glob
import pickle
import json
import numpy as np
import torch


class IMUDatasetCollection:
    IMU_SUBPATH = 'stimulis'
    POS_SUBPATH = 'pos'
    VEL_SUBPATH = 'vel'
    ACC_SUBPATH = 'acc'

    IMU_PATTERN = "imu_{}.csv"
    POS_PATTERN = "cartesianPos_{}.csv"
    VEL_PATTERN = "cartesianVec_{}.csv"
    ACC_PATTERN = "cartesianAec_{}.csv"

    def __init__(self,
                 path_to_data,
                 imu_subpath: str = 'imu',
                 pos_subpath: str = 'pos',
                 vel_subpath: str = 'vel',
                 acc_subpath: str = 'acc',
                 imu_pattern: str = "imu_{}.csv",
                 pos_pattern: str = "cartesianPos_{}.csv",
                 vel_pattern: str = "cartesianVec_{}.csv",
                 acc_pattern: str = "cartesianAec_{}.csv"
                 ) -> None:
        self.path_to_data = path_to_data
        self.POS_PATTERN = pos_pattern
        self.IMU_PATTERN = imu_pattern
        self.VEL_PATTERN = vel_pattern
        self.ACC_PATTERN = acc_pattern

        self.POS_SUBPATH = pos_subpath
        self.IMU_SUBPATH = imu_subpath
        self.VEL_SUBPATH = vel_subpath
        self.ACC_SUBPATH = acc_subpath

        self.imu_list = list(
            zip(*[
                glob.glob(os.path.join(self.path_to_data, self.IMU_SUBPATH, self.IMU_PATTERN.format('*')))
            ]))
        self.pos_list = list(
            zip(*[
                glob.glob(os.path.join(self.path_to_data, self.POS_SUBPATH, self.POS_PATTERN.format('*')))
            ]))
        self.vel_list = list(
            zip(*[
                glob.glob(os.path.join(self.path_to_data, self.VEL_SUBPATH, self.VEL_PATTERN.format('*')))
            ]))
        self.acc_list = list(
            zip(*[
                glob.glob(os.path.join(self.path_to_data, self.ACC_SUBPATH, self.ACC_PATTERN.format('*')))
            ]))

        self.length = len(self.imu_list)

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> Dict:
        return {
            'imu': pd.read_csv(os.path.join(self.path_to_data, self.IMU_SUBPATH, self.IMU_PATTERN.format(index))),
            'pos': pd.read_csv(os.path.join(self.path_to_data, self.POS_SUBPATH, self.POS_PATTERN.format(index))),
            'vel': pd.read_csv(os.path.join(self.path_to_data, self.VEL_SUBPATH, self.VEL_PATTERN.format(index))),
            'acc': pd.read_csv(os.path.join(self.path_to_data, self.ACC_SUBPATH, self.ACC_PATTERN.format(index))),
        }


class IMUDataset(Dataset):
    def __init__(self, path_to_data: str, features: List[str], target: str = 'pos') -> None:
        super().__init__()
        self.path_to_data = path_to_data
        with open(os.path.join(self.path_to_data, 'meta.json')) as f:
            self.meta = json.load(f)
        self.data = {}
        self.features = features
        self.target = target
        self.load_all()

    def load_all(self):
        for filename in self.meta['filenames']:
            with open(os.path.join(self.path_to_data, filename), 'rb') as f:
                self.data[filename] = pickle.load(f)

    def __getitem__(self, index):
        _, filename, local_index = self.meta['index_map'][index]
        _stimulis = np.hstack([self.data[filename]['imu'][item][local_index:local_index + self.meta['window_sz']] for item in self.features]).T
        _label = self.data[filename]['robot'][self.target][local_index + self.meta['window_sz'] - 1] - self.data[filename]['robot'][self.target][local_index]

        return torch.tensor(_stimulis, dtype=torch.float32), torch.tensor(_label, dtype=torch.float32)

    def __len__(self):
        return self.meta['len']


class IMUDatasetACC2ACC(Dataset):
    flange = torch.tensor([[0.7071, 0.7071, 0], [-0.7071, -0.7071, 0], [0, 0, 1]])

    def __init__(self, path_to_data: str) -> None:
        super().__init__()
        self.path_to_data = path_to_data
        with open(os.path.join(self.path_to_data, 'meta.json')) as f:
            self.meta = json.load(f)
        self.data = {}
        self.load_all()

    def load_all(self):
        for filename in self.meta['filenames']:
            with open(os.path.join(self.path_to_data, filename), 'rb') as f:
                self.data[filename] = pickle.load(f)

    def __getitem__(self, index):
        _, filename, local_index = self.meta['index_map'][index]
        _stimulis = self.data[filename]['imu']['acc'][local_index:local_index + self.meta['window_sz']]
        _stimulis_ts = torch.tensor(_stimulis, dtype=torch.float32)

        _label = self.data[filename]['robot']['acc'][local_index:local_index + self.meta['window_sz']]
        _label_ts = torch.tensor(_label, dtype=torch.float32)
        _label_ts[:, 2] -= 1

        _rot = self.data[filename]['robot']['rot'][local_index:local_index + self.meta['window_sz']]
        _rot_ts = torch.tensor(_rot, dtype=torch.float32)

        _label_ts = torch.matmul(torch.linalg.inv(_rot_ts), _label_ts.unsqueeze(-1)).squeeze(-1)
        _label_tf = torch.matmul(self.flange, _label_ts.unsqueeze(-1)).squeeze(-1)
        # _label_ts = _label_ts[:, [2, 0, 1]]

        return torch.tensor(_stimulis, dtype=torch.float32).T, torch.tensor(_label, dtype=torch.float32).T

    def __len__(self):
        return self.meta['len']


if __name__ == '__main__':
    ds = IMUDataset('./data_interp', features=['acc', 'gyro', 'mag'])
    ds.load_all()
    print(ds)
    x = ds[0]
    print(x)
