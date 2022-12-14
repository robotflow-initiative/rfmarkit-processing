from helpers import parse_record
import numpy as np

if __name__ == '__main__':

    record_name = '2021-11-13-19:56:08_cu.usbserial-124440_IMU_record.json'
    accel_raw, rpy, gyro, mag, pose_mat, timestamp = parse_record(record_name)
    length = len(accel_raw)

    res = {
        'id': np.repeat(np.array([['000000000000']]), length, axis=0),
        'timestamp': timestamp,
        'accel_x': accel_raw[:, 0],
        'accel_y': accel_raw[:, 1],
        'accel_z': accel_raw[:, 2],
        'gyro_x': gyro[:, 0],
        'gyro_y': gyro[:, 1],
        'gyro_z': gyro[:, 2],
        'mag_x': mag[:, 0],
        'mag_y': mag[:, 1],
        'mag_z': mag[:, 2],
        'pitch': rpy[:, 0],
        'roll': rpy[:, 1],
        'yaw': rpy[:, 2],
    }

    np.savez(f'imu_0000000000.npz', **res)