import json
import math

import numpy as np
import scipy
import transforms3d as t3d
import matplotlib.pyplot as plt
from scipy import signal

from typing import List, Any, Dict, Union, Tuple

from helpers import vectorize_to_np



class IMUAlgorithm(object):
    GRAVITY_NORM: float = 9.7964

    def __init__(self) -> None:
        pass

    @classmethod
    def rpy_to_pose_mat_np(cls, rpy_data: np.ndarray) -> np.ndarray:
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

    @classmethod
    def visualize_3d(cls, data: np.ndarray, timestamp: np.ndarray, title: str):
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

    @classmethod
    def visualize_1d(cls, data: np.ndarray, timestamp: np.ndarray, title: str):
        fig = plt.figure(figsize=(10, 8))

        ax = fig.add_subplot(111)
        ax.scatter(timestamp, data, s=2)
        ax.set_title(title)

        plt.show()

    @classmethod
    def filter_bandpass(cls,
                        data: np.ndarray,
                        band: Tuple[float, float] = (0.001, 8),
                        order: int = 1,
                        sample_freq: float = 100):
        res: np.array = np.copy(data)
        b, a = signal.butter(order, tuple(map(lambda x: 2 * x / sample_freq, band)), 'bandpass')
        for i in range(res.shape[-1]):
            res[..., i] = signal.filtfilt(b, a, data[..., i])

        # FIXME: Experimental
        # res -= np.linspace(res[0,:],np.array([0,0,0]), len(res))
        return res

    @classmethod
    def filter_middle(cls, data: np.ndarray, windows_sz: int = 5):
        res = np.copy(data)
        for idx in range(len(data) - windows_sz + 1):
            res[idx:idx + windows_sz] = np.repeat(np.expand_dims(np.mean(res[idx:idx + windows_sz], axis=0), axis=0),
                                                  windows_sz,
                                                  axis=0)
        return res

    # @classmethod
    # def get_measured_gravity(cls, accel, pose_mat, thresh=150):
    #     return ((pose_mat[:thresh].transpose(0, 2, 1) @ accel[:thresh][:, :, None])[:, :, 0]).mean(axis=0) #* np.array([0., 0., 1.])

    # @classmethod
    # def get_accel_offset(accel: np.ndarray, g: np.ndarray, thresh: int = 100) -> np.ndarray:
    #     reading = np.mean(accel[:thresh, :], axis=0)
    #     offset = reading - g
    #     return offset

    @classmethod
    def unpack_npz(cls, npzfile: np.ndarray, trim_thresh: int = 000, **kwargs):
        accel_i = np.squeeze(np.stack([npzfile['accel_x'], -1 * npzfile['accel_y'], npzfile['accel_z']], axis=1)).astype(
            np.float64) * cls.GRAVITY_NORM
        gyro = np.squeeze(np.stack([npzfile['gyro_x'], -1 * npzfile['gyro_y'], npzfile['gyro_z']], axis=1)).astype(np.float64)
        rpy = np.squeeze(np.stack([npzfile['roll'], npzfile['pitch'], npzfile['yaw']], axis=1)).astype(np.float64) * np.pi / 180
        mag = np.squeeze(-np.stack([npzfile['mag_x'], -1 * npzfile['mag_y'], npzfile['mag_z']], axis=1)).astype(np.float64)
        timestamp = npzfile['timestamp'].astype(np.float64)
        pose_mat = cls.rpy_to_pose_mat_np(rpy).astype(np.float64)

        # Trim
        if trim_thresh <= 0:
            trim_thresh = np.where(np.squeeze(npzfile['uart_buffer_len']) < 10)[0].min()

        # FIXME Experimental
        accel_i = accel_i[trim_thresh:]
        gyro = gyro[trim_thresh:]
        rpy = rpy[trim_thresh:]
        mag = mag[trim_thresh:]
        pose_mat = pose_mat[trim_thresh:]
        timestamp = timestamp[trim_thresh:]
        return {'accel_i': accel_i, 'gyro': gyro, 'mag': mag, 'rpy': rpy, 'pose_mat': pose_mat, 'timestamp': timestamp}

    @classmethod
    def substract_gravity(cls,
                          accel_i,
                          rpy,
                          timestamp,
                          pose_mat,
                          measurement_bias: np.array = np.array([0, 0, 0], dtype=np.float64),
                          **kwargs):

        accel_i -= measurement_bias
        accel_w = np.empty_like(accel_i)
        # Project gravity to local coordinate,
        # then substract accel initial readings (mesured g) with projected gravity
        # TODO: try assumed_gain = np.array([1,1,1])

        GRAVITY_SHANGHAI = np.array([0, 0, -cls.GRAVITY_NORM])

        # for i in range(len(timestamp)):
        #     accel_w[i] = pose_mat[i].T @ accel_i[i] - GRAVITY_SHANGHAI

        # Sustract gravity
        gravity_i = np.empty_like(accel_i)
        for i in range(len(timestamp)):
            gravity_i[i] = pose_mat[i] @ GRAVITY_SHANGHAI

        for i in range(len(timestamp)):
            accel_w[i] = pose_mat[i].T @ (accel_i[i] - gravity_i[i])

        print(f'accel_w.mean={np.mean(accel_w,axis=0)}')

        # filter accel
        # accel = cls.filter_accel(accel, (0.005,0.999))

        return {'accel_w': accel_w, 'gravity_i': gravity_i}

    @classmethod
    def zero_vel_determination(cls, gyro: np.ndarray, accel_i: np.ndarray, thresh: Tuple[float] = (1, np.inf, np.inf, np.inf)) -> bool:
        if gyro.shape[0] > 0 and accel_i.shape[0] > 0:
            gyro_mean = np.sqrt(np.sum(np.mean(gyro, axis=0)**2))
            gyro_std = np.mean(np.std(gyro, axis=0))
            accel_mean = np.sqrt(np.sum(np.mean(accel_i, axis=0)**2))
            accel_std = np.mean(np.std(accel_i, axis=0))
            # print(vel_mean, gyro_mean, gyro_std, accel_mean, accel_std)
            if all([gyro_mean < thresh[0], gyro_std < thresh[1], accel_mean < thresh[2], accel_std < thresh[3]]):
                return True
            else:
                # print(f"gyro.mean={gyro_mean},.std={gyro_std};accel.mean={accel_mean},.std={accel_std}")
                return False
        else:
            return False

    @classmethod
    def run_zerovel_detection(cls, accel_i, gyro, rpy, timestamp, window_sz: int = 5, **kwargs):
        # calc velocity, with zero velocity update policy
        zero_vel = np.zeros_like(timestamp, dtype=np.int16)
        cali_points = []

        # TODO: Filter acceleration
        accel_i_filterd = cls.filter_bandpass(accel_i)
        cls.visualize_3d(accel_i, timestamp,'accel_not_filtered')
        cls.visualize_3d(accel_i_filterd, timestamp,'accel_filtered')

        for idx in range(len(timestamp)):
            if cls.zero_vel_determination(gyro[idx - window_sz:idx, :], accel_i_filterd[idx - window_sz:idx, :]):
                zero_vel[idx] = 1
                cali_points.append({
                    'idx': idx,
                    'mes': accel_i[idx],
                    'rpy': rpy[idx],
                    'vel': np.array([0, 0, 0], dtype=np.float64)
                })
        zero_vel[-1] = 1
        cali_points.append({
                    'idx': len(zero_vel) - 1,
                    'mes': accel_i[-1],
                    'rpy': rpy[-1],
                    'vel': np.array([0, 0, 0], dtype=np.float64)
                })
        return {'zero_vel': zero_vel, 'cali_points': cali_points}

    @classmethod
    def get_accel_compensation(cls, cali_points, gravity_i,
                           **kwargs) -> Tuple[Union[None, np.array], Union[None, List[Dict[str, Any]]]]:
        if (len(cali_points) <= 0):
            return None

        mes = np.zeros(shape=(len(cali_points), 3))
        real = np.zeros(shape=(len(cali_points), 3))
        for idx, point in enumerate(cali_points):
            mes[idx] = point['mes']
            real[idx] = gravity_i[point['idx']]
        # Plan1 mes = real + bias + noise

        # bias of accel_i
        bias = mes.mean(axis=0) - real.mean(axis=0)
        print(f"accel_i.bias={bias}")

        return bias

    @classmethod
    def get_vel_compensation(cls, vel: np.array, cali_points: List[Dict[str, Any]]):
        vel_offset = np.zeros_like(vel)
        last_point = {'idx': 0, 'vel': np.array([0, 0, 0], dtype=np.float64)}
        for point in cali_points:
            vel_offset[last_point['idx']:point['idx']] = np.linspace(vel[last_point['idx']] - last_point['vel'],
                                                                     vel[point['idx']] - point['vel'],
                                                                     point['idx'] - last_point['idx'])
                                                                    
            last_point = point

        return vel_offset

    @classmethod
    def run_vel_construction(cls, accel_w, timestamp, **kwargs):
        # calc velocity, with zero velocity update policy

        vel = np.zeros_like(accel_w)
        for i in range(len(timestamp) - 1):
            # Measured acceleration is inverse of actural acceleration
            vel[i + 1] = vel[i] - 0.5 * (accel_w[i + 1] + accel_w[i]) * (timestamp[i + 1] - timestamp[i])
        return {'vel': vel}

    @classmethod
    def run_pos_construction(cls, vel, timestamp, **kwargs):
        # calc displacement
        # Mid-value integration
        pos = np.zeros_like(vel)
        for i in range(len(timestamp) - 1):
            pos[i + 1] = pos[i] + 0.5 * (vel[i + 1] + vel[i]) * (timestamp[i + 1] - timestamp[i])
        return {'pos': pos}

    @classmethod
    def reconstruct(cls, ctx: Dict[str, np.array] = None, measurement_filepath: str = ''):
        if ctx is None:
            ctx: Dict[str, np.array] = cls.unpack_npz(np.load(measurement_filepath))  # e.g. ./imu_abcdef123456.npz
        print(ctx.keys())

        # # Filter accel_i
        # accel_i = cls.filter_accel_middle(ctx['accel_i'])
        # ctx['accel_i'] = accel_i

        # Calibrate accel, get accel_w and gravity_i
        # ctx['accel_i'] = cls.filter_bandpass(ctx['accel_i'])
        
        ctx = {**ctx, **cls.substract_gravity(ctx['accel_i'], 
                                              ctx['rpy'], 
                                              ctx['timestamp'], 
                                              ctx['pose_mat'])}

        ctx['accel_w'] = cls.filter_bandpass(ctx['accel_w'])

        # get zero_vel vector and cali_points
        ctx = {**ctx, **cls.run_zerovel_detection(ctx['accel_i'], 
                                                  ctx['gyro'], 
                                                  ctx['rpy'],
                                                  ctx['timestamp'])}

        accel_i_bias = cls.get_accel_compensation(ctx['cali_points'], 
                                                  ctx['gravity_i'])
        if accel_i_bias is not None:
            ctx['accel_i'] = ctx['accel_i'] - accel_i_bias
            # Re-run steps using the calibrated accel
            ctx.update(**cls.substract_gravity(ctx['accel_i'], 
                                              ctx['rpy'], 
                                              ctx['timestamp'], 
                                              ctx['pose_mat']))
            ctx['accel_w'] = cls.filter_bandpass(ctx['accel_w'])
            
            ctx = {**ctx, **cls.run_zerovel_detection(ctx['accel_i'], 
                                                  ctx['gyro'], 
                                                  ctx['rpy'],
                                                  ctx['timestamp'])}

            # ctx['accel_w'] = cls.filter_bandpass(ctx['accel_w'])
            ctx = {**ctx, **cls.run_vel_construction(ctx['accel_w'], ctx['timestamp'])}
            vel_offset = cls.get_vel_compensation(ctx['vel'], 
                                                  ctx['cali_points'])
            cls.visualize_3d(ctx['vel'] , ctx['timestamp'],'vel_not_compensated')
            ctx['vel'] -= vel_offset
            cls.visualize_3d(ctx['vel'] , ctx['timestamp'],'vel_compensated')
            ctx = {**ctx, **cls.run_pos_construction(ctx['vel'], ctx['timestamp'])}
            
        else:
            ctx = {**ctx, **cls.run_vel_construction(ctx['accel_w'], ctx['timestamp'])}
            ctx = {**ctx, **cls.run_pos_construction(ctx['vel'], ctx['timestamp'])}

        return ctx
