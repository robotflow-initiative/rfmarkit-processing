import numpy as np
import matplotlib.pyplot as plt
import json
import scipy
from typing import List, Any, Dict, Union
import transforms3d as t3d
from scipy import signal
import math


def vectorize_to_np(record_list: List[Dict[str, Any]], keys: List[str]) -> Dict[str, np.ndarray]:
    """Vectorizing record

    Args:
        record_list (List[Dict[str, Any]]): List of records, each record is a bundled dictionary
        keys (List[str]): keys to extract from records

    Returns:
        Dict[str, np.ndarray]: A dictionary in which keys are desired and values are numpy arrays
    """
    assert len(keys) > 0
    assert len(record_list) > 0
    res = {}
    for key in keys:
        res[key] = np.expand_dims(np.array([record[key] for record in record_list]), axis=-1)

    # Verify length
    _length: int = len(res[keys[0]])
    for key in keys:
        if _length != len(res[key]):
            raise ValueError("Not every attribute has the same length")

    return res


def rpy_to_pose_mat_np(rpy_data: np.ndarray) -> np.ndarray:
    """Convert roll-pitch-yaw data to transform matrix

    Args:
        ryp_data (np.ndarray): 2-D matrix
        [[r0,p0,y0],[r1,p1,y1],...]

    Returns:
        np.ndarray: [description]
    """
    length = rpy_data.shape[0]
    pose_mat = np.empty(shape=(length, 3, 3))
    for idx in range(length):
        pose_mat[idx] = t3d.euler.euler2mat(*rpy_data[idx], 'rxyz')
    return pose_mat


def parse_record(path: str, g: float = 9.8):
    c = np.pi / 180  # deg->rad conversion
    # use json module to load
    _record_list = json.load(open(path))
    _res = x = vectorize_to_np(_record_list, [
        'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'pitch', 'roll', 'yaw', 'time', 'mag_x', 'mag_y', 'mag_z'
    ])
    _accel = np.hstack([_res['accel_x'], _res['accel_y'], _res['accel_z']])
    _rpy = np.hstack([_res['roll'], _res['pitch'], _res['yaw']])
    _gyro = np.hstack([_res['gyro_x'], _res['gyro_y'], _res['gyro_z'] * c])
    _mag = np.hstack([_res['mag_x'], _res['mag_y'], _res['mag_z']])
    _pose_mat = rpy_to_pose_mat_np(_rpy)
    _timestamp = _res['time']
    return _accel, _rpy, _gyro, _mag, _pose_mat, _timestamp


def load_data_from_json(pth_to_json) -> Dict[str, np.array]:
    record_name = pth_to_json  # path-like string
    c = 1. / 180. * np.pi  # deg->rad conversion

    # use json module to load
    record_list = json.load(open(record_name))
    # record_list = record_list[100:]
    _data = vectorize_to_np(record_list, [
        'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'pitch', 'roll', 'yaw', 'time'
    ])
    accel = np.hstack([_data['accel_x'], _data['accel_y'], _data['accel_z']])
    rpy = np.hstack([_data['roll'] * c, _data['pitch'] * c, _data['yaw'] * c])
    gyro = np.hstack([_data['gyro_x'], _data['gyro_y'], _data['gyro_z']])
    mag = np.hstack([_data['mag_x'], _data['mag_y'], _data['mag_z']])
    pose_mat = rpy_to_pose_mat_np(rpy)
    timestamp = _data['time']

    return {'accel': accel, 'rpy': rpy, 'gyro': gyro, 'mag': mag, 'pose_mat': pose_mat, 'timestamp': timestamp}


def linear_map(samples: np.ndarray, timestamp: np.ndarray, rate):
    t_delta = 1 / rate
    t_max = int(timestamp.max())
    new_timestamp = np.linspace(0, t_max, t_max * rate + 1)
    samples


def visualize_3d(data: np.ndarray, timestamp: np.ndarray, title: str):
    fig = plt.figure(figsize=(32, 8))

    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=timestamp)
    ax.set_xlabel(title + '-X', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel(title + '-Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel(title + '-Z', fontdict={'size': 15, 'color': 'red'})

    ax = fig.add_subplot(122)
    ax.scatter(timestamp, data[:, 0], s=2, c='r')
    ax.scatter(timestamp, data[:, 1], s=2, c='g')
    ax.scatter(timestamp, data[:, 2], s=2, c='b')
    ax.set_title(title)

    plt.show()


def visualize_1d(data: np.ndarray, timestamp: np.ndarray, title: str):
    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111)
    ax.scatter(timestamp, data, s=2)
    ax.set_title(title)

    plt.show()