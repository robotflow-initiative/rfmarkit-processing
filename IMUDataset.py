from threading import local
from typing import List, Tuple
from torch.utils.data import Dataset
import re
from typing import Tuple
import pandas as pd
import os
import glob
import pickle
import json
import numpy as np
import torch


class IMUDatasetCollection:
    LABEL_SUBPATH = 'label'
    STIMULIS_SUBPATH = 'stimulis'
    LABEL_PATTERN = ["cartesianPos_{}.csv", "cartesianVec_{}.csv", "cartesianAec_{}.csv"]
    STIMULIS_PATTERN = ["imu_{}.csv"]

    def __init__(self,
                 path_to_data,
                 label_subpath: str = 'label',
                 stimulis_subpath: str = 'stimulis',
                 label_pattern: List[str] = ["cartesianPos_{}.csv", "cartesianVec_{}.csv", "cartesianAec_{}.csv"],
                 stimulis_pattern: List[str] = ["imu_{}.csv"]) -> None:
        self.path_to_data = path_to_data
        self.LABEL_PATTERN = label_pattern
        self.STIMULIS_PATTERN = stimulis_pattern
        self.LABEL_SUBPATH = label_subpath
        self.STIMULIS_SUBPATH = stimulis_subpath

        self.label_list = list(
            zip(*[
                glob.glob(os.path.join(self.path_to_data, self.LABEL_SUBPATH, pattern.format('*')))
                for pattern in self.LABEL_PATTERN
            ]))
        self.stimulis_list = list(
            zip(*[
                glob.glob(os.path.join(self.path_to_data, self.STIMULIS_SUBPATH, pattern.format('*')))
                for pattern in self.STIMULIS_PATTERN
            ]))

        self.length = sum([
            len(glob.glob(os.path.join(self.path_to_data, self.STIMULIS_SUBPATH, pattern.format('*'))))
            for pattern in stimulis_pattern
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> Tuple:
        return {
            'label': [
                pd.read_csv(item) for item in map(
                    lambda x: os.path.join(self.path_to_data, self.LABEL_SUBPATH, x.format(index)), self.LABEL_PATTERN)
            ],
            'stimulis': [
                pd.read_csv(item) for item in map(
                    lambda x: os.path.join(self.path_to_data, self.STIMULIS_SUBPATH, x.format(index)), self.STIMULIS_PATTERN)
            ]
        }


class IMUDataset(Dataset):
    def __init__(self, path_to_data: str, features: List[str]) -> None:
        super().__init__()
        self.path_to_data = path_to_data
        with open(os.path.join(self.path_to_data, 'meta.json')) as f:
            self.meta = json.load(f)
        self.data = {}
        self.features = features
        self.load_all()

    def load_all(self):
        for filename in self.meta['filenames']:
            with open(os.path.join(self.path_to_data, filename), 'rb') as f:
                self.data[filename] = pickle.load(f)

    def __getitem__(self, index):
        _, filename, local_index = self.meta['index_map'][index]
        _stimulis = np.hstack([self.data[filename]['imu'][item][local_index:local_index+self.meta['window_sz']] for item in self.features]).T
        _label = self.data[filename]['robot']['pos'][local_index+self.meta['window_sz'] - 1] - self.data[filename]['robot']['pos'][local_index]

        return torch.tensor(_stimulis, dtype=torch.float32), torch.tensor(_label, dtype=torch.float32)

    def __len__(self):
        return self.meta['len']


if __name__ == '__main__':
    ds = IMUDataset('./data_interp', features=['acc', 'gyro', 'mag'])
    ds.load_all()
    print(ds)
    x = ds[0]
    print(x)