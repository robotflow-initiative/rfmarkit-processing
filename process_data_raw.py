from IMUDataset import IMUDatasetCollection
from algorithm import IMUAlgorithm
import numpy as np
import pickle
import tqdm
import os


def from_collection_entry_to_np(data_collection, index):
    imu_df = data_collection[index]['stimulis'][0]
    imu_np = imu_df.to_numpy()
    imu_ts = imu_np[:, 2].astype(np.float64)
    imu_ts -= imu_ts[0]
    imu_acc = imu_np[:, 3:6].astype(np.float64)
    imu_gyro = imu_np[:, 6:9].astype(np.float64)
    imu_mag = imu_np[:, 9:12].astype(np.float64)
    imu_euler = imu_np[:, 12:15].astype(np.float64)
    imu_quat = imu_np[:, 15:19].astype(np.float64)
    imu_rot = IMUAlgorithm.quat_to_pose_mat_np(imu_quat)

    robot_df = data_collection[index]['label'][0]
    robot_np = robot_df.to_numpy()
    # @remark Resolution 1ms
    robot_ts = robot_np[:, 1].astype(np.float64) * 1e-3
    robot_ts -= robot_ts[0]
    robot_pos = robot_np[:, 14:17].astype(np.float64)
    robot_rot = np.stack([robot_np[:, 2:5], robot_np[:, 6:9], robot_np[:, 10:13]], axis=-1)
    robot_quat = IMUAlgorithm.pose_mat_to_quat_np(robot_rot)

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
    robot_quat_interp = np.stack([
        np.interp(global_ts, robot_data['ts'], robot_data['quat'][:, 0]),
        np.interp(global_ts, robot_data['ts'], robot_data['quat'][:, 1]),
        np.interp(global_ts, robot_data['ts'], robot_data['quat'][:, 2]),
        np.interp(global_ts, robot_data['ts'], robot_data['quat'][:, 3]),
    ],
                                 axis=-1)
    robot_rot_interp = IMUAlgorithm.quat_to_pose_mat_np(robot_quat_interp)

    return {
        'interp': {
            'imu': {
                'acc': imu_acc_interp,
                'gyro': imu_gyro_interp,
                'mag': imu_mag_interp,
                'euler': imu_euler_interp,
                'quat': imu_quat_interp,
                'rot': imu_rot_interp,
            },
            'robot': {
                'pos': robot_pos_interp,
                'quat': robot_quat_interp,
                'rot': robot_rot_interp,
            }
        }
    }

def work(data_collection, index, output_dir):
    res = from_collection_entry_to_np(data_collection, index)
    res_interp = interp_data(res['imu'], res['robot'])
    with open(os.path.join(output_dir, 'record_{0:06}.pkl'.format(index)), 'wb') as f:
        pickle.dump(res_interp, f)

if __name__ == '__main__':
    data_collection = IMUDatasetCollection('./data_raw',
                                           label_subpath='Pos',
                                           stimulis_subpath='IMU',
                                           label_pattern=['cartesianPos_{}.csv'],
                                           stimulis_pattern=["imu_{}.csv"])
    output_dir = './data_interp'

    from multiprocessing import Pool
    pool = Pool(8)
    meta_fp = open(os.path.join(output_dir, 'meta.txt'), 'w')
    with tqdm.tqdm(range(len(data_collection))) as pbar:
        for index in range(1, 1 + len(data_collection)):
            pool.apply_async(func=work, args=(data_collection, index, output_dir,), callback=lambda res: pbar.update(), error_callback=lambda err: print(err))
            meta_fp.write('record_{0:06}.pkl\n'.format(index))
        pool.close()
        pool.join()
    meta_fp.close()