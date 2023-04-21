import json
from typing import Dict
from IMUDataset import IMUDatasetCollection
from algorithm import IMUAlgorithm
import numpy as np
import pickle
import tqdm
import os


def from_collection_entry_to_np(data_collection, index, offset=0):
    """Load Numpy format data from data collection

    Args:
        data_collection ([type]): [description]
        index ([type]): [description]
        offset (int, optional): Offset of IMU data relative to Robot, positive, IMU[t+offset] = Robot[t]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    data_entry = data_collection[index]
    imu_df = data_entry['imu'][offset:]
    imu_np = imu_df.to_numpy()
    imu_ts = imu_np[:, 2].astype(np.float64)
    imu_ts -= imu_ts[0]
    imu_acc = imu_np[:, 3:6].astype(np.float64)
    imu_gyro = imu_np[:, 6:9].astype(np.float64)
    imu_mag = imu_np[:, 9:12].astype(np.float64)
    imu_euler = imu_np[:, 12:15].astype(np.float64)
    imu_quat = imu_np[:, 15:19].astype(np.float64)
    imu_rot = IMUAlgorithm.quat_to_pose_mat_np(imu_quat)

    robot_pos_df = data_entry['pos'][:len( data_entry['pos'])-offset]
    robot_pos_np = robot_pos_df.to_numpy()
    # @remark Resolution 1ms
    robot_ts = robot_pos_np[:, 1].astype(np.float64) * 1e-3
    robot_ts -= robot_ts[0]
    robot_pos = robot_pos_np[:, 14:17].astype(np.float64)
    robot_rot = np.stack(
        [robot_pos_np[:, 2:5], robot_pos_np[:, 6:9], robot_pos_np[:, 10:13]], axis=-1)
    robot_quat = IMUAlgorithm.pose_mat_to_quat_np(robot_rot)
    # Diff method
    # robot_vel = np.zeros_like(robot_pos)
    # robot_vel[1:] = (robot_pos[1:] - robot_pos[:-1]) * 1e3
    # robot_vel = IMUAlgorithm.filter_middle(robot_vel, 200)

    robot_vel = data_entry['vel'][:len( data_entry['pos'])-offset]
    assert len(robot_vel) == len(robot_pos)
    robot_vel_np = robot_vel.to_numpy()
    robot_vel = robot_vel_np[:, 2:5].astype(np.float64)
    robot_vel_ang = robot_vel_np[:, 5:8].astype(np.float64)

    robot_acc = data_entry['acc'][:len( data_entry['pos'])-offset]
    assert len(robot_acc) == len(robot_pos)
    robot_acc_np = robot_acc.to_numpy()
    robot_acc = robot_acc_np[:, 2:5].astype(np.float64)
    robot_acc_ang = robot_acc_np[:, 5:8].astype(np.float64)

    return {
        'imu': {
            'ts': imu_ts,
            'acc': imu_acc,
            'gyro': imu_gyro,
            'mag': imu_mag,
            'euler': imu_euler,
            'quat': imu_quat,
            'rot': imu_rot,
        },
        'robot': {
            'ts': robot_ts,
            'vel': robot_vel,
            'vel_ang': robot_vel_ang,
            'acc': robot_acc,
            'acc_ang': robot_acc_ang,
            'pos': robot_pos,
            'quat': robot_quat,
            'rot': robot_rot,
        }
    }


global_dt = 5e-3


def interp_data(imu_data, robot_data):
    global global_dt
    n_ticks = int(robot_data['ts'][-1] / global_dt)
    last_t = n_ticks * global_dt
    global_ts = np.linspace(0, last_t - global_dt, n_ticks)

    imu_acc_interp = np.stack([
        np.interp(global_ts, imu_data['ts'], imu_data['acc'][:, 0]),
        np.interp(global_ts, imu_data['ts'], imu_data['acc'][:, 1]),
        np.interp(global_ts, imu_data['ts'], imu_data['acc'][:, 2])
    ],
        axis=-1)
    imu_gyro_interp = np.stack([
        np.interp(global_ts, imu_data['ts'], imu_data['gyro'][:, 0]),
        np.interp(global_ts, imu_data['ts'], imu_data['gyro'][:, 1]),
        np.interp(global_ts, imu_data['ts'], imu_data['gyro'][:, 2])
    ],
        axis=-1)
    imu_mag_interp = np.stack([
        np.interp(global_ts, imu_data['ts'], imu_data['mag'][:, 0]),
        np.interp(global_ts, imu_data['ts'], imu_data['mag'][:, 1]),
        np.interp(global_ts, imu_data['ts'], imu_data['mag'][:, 2])
    ],
        axis=-1)
    imu_euler_interp = np.stack([
        np.interp(global_ts, imu_data['ts'], imu_data['euler'][:, 0]),
        np.interp(global_ts, imu_data['ts'], imu_data['euler'][:, 1]),
        np.interp(global_ts, imu_data['ts'], imu_data['euler'][:, 2])
    ],
        axis=-1)
    imu_quat_interp = np.stack([
        np.interp(global_ts, imu_data['ts'], imu_data['quat'][:, 0]),
        np.interp(global_ts, imu_data['ts'], imu_data['quat'][:, 1]),
        np.interp(global_ts, imu_data['ts'], imu_data['quat'][:, 2]),
        np.interp(global_ts, imu_data['ts'], imu_data['quat'][:, 3]),
    ],
        axis=-1)
    imu_rot_interp = IMUAlgorithm.quat_to_pose_mat_np(imu_quat_interp)

    robot_pos_interp = np.stack([
        np.interp(global_ts, robot_data['ts'], robot_data['pos'][:, 0]),
        np.interp(global_ts, robot_data['ts'], robot_data['pos'][:, 1]),
        np.interp(global_ts, robot_data['ts'], robot_data['pos'][:, 2]),
    ],
        axis=-1)
    robot_vel_interp = np.stack([
        np.interp(global_ts, robot_data['ts'], robot_data['vel'][:, 0]),
        np.interp(global_ts, robot_data['ts'], robot_data['vel'][:, 1]),
        np.interp(global_ts, robot_data['ts'], robot_data['vel'][:, 2]),
    ],
        axis=-1)
    robot_vel_ang_interp = np.stack([
        np.interp(global_ts, robot_data['ts'], robot_data['vel_ang'][:, 0]),
        np.interp(global_ts, robot_data['ts'], robot_data['vel_ang'][:, 1]),
        np.interp(global_ts, robot_data['ts'], robot_data['vel_ang'][:, 2]),
    ],
        axis=-1)
    robot_acc_interp = np.stack([
        np.interp(global_ts, robot_data['ts'], robot_data['acc'][:, 0]),
        np.interp(global_ts, robot_data['ts'], robot_data['acc'][:, 1]),
        np.interp(global_ts, robot_data['ts'], robot_data['acc'][:, 2]),
    ],
        axis=-1)

    robot_acc_ang_interp = np.stack([
        np.interp(global_ts, robot_data['ts'], robot_data['acc_ang'][:, 0]),
        np.interp(global_ts, robot_data['ts'], robot_data['acc_ang'][:, 1]),
        np.interp(global_ts, robot_data['ts'], robot_data['acc_ang'][:, 2]),
    ],
        axis=-1)

    robot_quat_interp = np.stack([
        np.interp(global_ts, robot_data['ts'], robot_data['quat'][:, 0]),
        np.interp(global_ts, robot_data['ts'], robot_data['quat'][:, 1]),
        np.interp(global_ts, robot_data['ts'], robot_data['quat'][:, 2]),
        np.interp(global_ts, robot_data['ts'], robot_data['quat'][:, 3]),
    ],
        axis=-1)
    robot_rot_interp = IMUAlgorithm.quat_to_pose_mat_np(robot_quat_interp)

    return {
        'imu': {
            'ts': global_ts,
            'acc': imu_acc_interp,
            'gyro': imu_gyro_interp,
            'mag': imu_mag_interp,
            'euler': imu_euler_interp,
            'quat': imu_quat_interp,
            'rot': imu_rot_interp,
        },
        'robot': {
            'ts': global_ts,
            'vel': robot_vel_interp,
            'vel_ang': robot_vel_ang_interp,
            'acc': robot_acc_interp,
            'acc_ang': robot_acc_ang_interp,
            'pos': robot_pos_interp,
            'quat': robot_quat_interp,
            'rot': robot_rot_interp,
        },
        'len': len(global_ts)
    }


def get_imu_robot_offset(res: Dict[str, Dict[str, np.ndarray]]):
    SEARCH_T_MAX: int = 1000
    SAMPLE_LENGTH: int = 500
    imu_sample = res['imu']['quat'][:SAMPLE_LENGTH]
    robot_sample = res['robot']['quat'][:SEARCH_T_MAX+SAMPLE_LENGTH]
    errors: np.ndarray = ((np.lib.stride_tricks.as_strided(
        robot_sample, (SAMPLE_LENGTH, 4)) - imu_sample)**2).sum(axis=1)
    offset: int = int(errors.argmin())  # IMU[t+offset] = Robot[t]
    return offset


def work(data_collection, index, output_dir):
    res = from_collection_entry_to_np(data_collection, index)
    res_interp = interp_data(res['imu'], res['robot'])
    with open(os.path.join(output_dir, 'record_{0:06}.pkl'.format(index)), 'wb') as f:
        pickle.dump(res_interp, f)


if __name__ == '__main__':
    data_collection = IMUDatasetCollection('/hdd0/data/imu_data/N_1-1925',
                                           imu_subpath='imu',
                                           pos_subpath='Pos',
                                           vel_subpath='Vec',
                                           acc_subpath="Aec")
    output_dir = './data_interp'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    WINDOW_SZ = 200

    record_index_map = []
    record_index_max: int = 0
    record_filenames = []

    with tqdm.tqdm(range(len(data_collection))) as pbar:
        for index in range(1, 1 + len(data_collection)):
            offset: int = 0
            try:
                res = from_collection_entry_to_np(
                    data_collection, index)  # 读取数据，csv->numpy
                offset = get_imu_robot_offset(res)  # 计算IMU相对Robot的滞后
                res = from_collection_entry_to_np(
                    data_collection, index, offset)  # 重新读取
            except Exception as err:
                print(err)
                continue
            res_interp = interp_data(res['imu'], res['robot'])  # 差值
            res_filename = 'record_{0:06}.pkl'.format(index)  # 计算文件名
            with open(os.path.join(output_dir, res_filename), 'wb') as f:
                pickle.dump(res_interp, f)  # 保存成pickle

            res_dataset_len = res_interp['len'] - WINDOW_SZ + 1  # 数据集的长度等于记录数量减去窗长加一
            record_index_map.extend([(record_index_max + local_index, res_filename, local_index)
                                     for local_index in range(res_dataset_len)])  # 建立index_map，将单调增的序号映射到每一份独立记录和记录本地的偏移
            record_index_max += res_dataset_len  # 更新最大记录序号的值

            record_filenames.append(res_filename)
            pbar.set_description(f"Index={index}, Offset={offset}")
            pbar.update()

    with open(os.path.join(output_dir, 'meta.json'), 'w+') as meta_fp:
        meta = {
            'window_sz': WINDOW_SZ,
            'len': len(record_index_map),
            'filenames': record_filenames,
            'index_map': record_index_map,
        }
        json.dump(meta, meta_fp, indent=4)
