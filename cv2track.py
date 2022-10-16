import glob
import json
import math
import os
import time
from collections import Counter
from typing import List, Dict
from typing import Tuple

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy import signal


class TrackSequence:
    def __init__(self, center, n_step: int = 3):
        # center: {"x": float
        #          "y": float
        #          "r": float
        #          "t": float
        #          }, ...
        self.seq = [center]
        self.n_step: int = n_step
        self.active: int = n_step
        self.idx = -1

    @property
    def center(self):
        return self.seq[-1]

    @property
    def is_active(self):
        return self.active > 0

    @property
    def duration(self):
        return self.seq[-1]["t"] - self.seq[0]["t"]

    def __sub__(self, other):
        return math.sqrt((other["x"] - self.seq[-1]["x"]) ** 2 + (other["y"] - self.seq[-1]["y"]) ** 2)

    def __len__(self):
        return len(self.seq)

    def append(self, c):
        self.seq.append(c)
        self.active = self.n_step

    def set(self):
        self.active = self.n_step

    def step(self):
        self.active -= 1

    @classmethod
    def decode_manchester_byte(cls, raw_bits: List[int]):
        """
        :param raw_bits: [1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1]
        :return: 0-255 value or -1 for failure
        """
        pattern = [1, -1]
        bits = []
        assert len(raw_bits) >= 20
        for i in range(8):
            sec = raw_bits[4 + 2 * i:4 + 2 * i + 2]
            if sec[0] == pattern[0] and sec[1] == pattern[1]:
                bits.append(0)
            elif sec[0] == pattern[1] and sec[1] == pattern[0]:
                bits.append(1)
                pattern = pattern[::-1]
            else:
                return -1
        result = sum([bits[i] * (2 ** (7 - i)) for i in range(8)])
        return result

    def process_track(self, frames):
        DIFF_THRESH = 800  # Difference in luminance between light/dark frames
        PERIOD = 0.05

        raw_lux = np.zeros(shape=(len(self),), dtype=float)
        raw_lux_dt = np.zeros(shape=(len(self),), dtype=float)
        for idx, center in enumerate(self.seq):
            x, y, r, t, frame_no = center['x'], center['y'], int(center['r']), center['t'], center['frame_no']
            raw_lux[idx] = np.sum(frames[frame_no][y - r:y + r, x - r:x + r])  # Get luminance
            raw_lux_dt[idx] = t  # Temporarily save t，this is not dt

        raw_lux_dt[1:] = raw_lux_dt[1:] - raw_lux_dt[:-1]  # Finally get frame time deltas
        raw_lux_dt[0] = 0  # Finally get frame deltas
        # Usually the delta is small
        frame_delta_avg, frame_delta_std = raw_lux_dt.mean(), raw_lux_dt.std()
        print(f"frame delta avg={frame_delta_avg}, std={frame_delta_std}")

        # Get frame luminance difference
        raw_lux_diff = np.zeros_like(raw_lux)
        raw_lux_diff[1:] = raw_lux[1:] - raw_lux[:-1]
        raw_lux_diff2 = np.zeros_like(raw_lux_diff)
        raw_lux_diff2[1:] = raw_lux_diff[1:] - raw_lux_diff[:-1]
        # Convert to binary using threshold
        # FIXME:
        # _thresh = abs(np.average(raw_lux_diff)) * 2
        _thresh = 800
        raw_lux_diff_binary = (raw_lux_diff > _thresh).astype(np.int64) - (raw_lux_diff < - _thresh).astype(
            np.int64)
        lux_filtered = TrackerUtils.filter_bandpass(np.expand_dims(raw_lux, 1), (8, 49.99), order=7, sample_freq=100)

        # Visualisation
        x_axis = np.linspace(0, self.duration, len(self))
        plt.plot(x_axis, raw_lux, linewidth=1)
        plt.title('raw_lux')
        plt.show()
        plt.plot(x_axis, lux_filtered, linewidth=1)
        plt.title('raw_filtered')
        plt.show()
        plt.plot(x_axis, raw_lux_diff, linewidth=1)
        plt.plot(x_axis, np.ones_like(x_axis) * _thresh, linewidth=1)
        plt.title('raw_lux_diff')
        plt.show()
        plt.plot(x_axis, raw_lux_diff2, linewidth=1)
        plt.title('raw_lux_diff2')
        plt.show()
        plt.plot(x_axis, raw_lux_diff_binary, linewidth=1)
        plt.title('raw_lux_diff_binary')
        plt.show()

        sync_interval = 0  # Length of sync period approx to 3 * PERIOD
        sync_started = False  # If the sync process is started, a.k.a luminous frame encountered
        value_interval = 0  # Interval between neighbor peaks
        state = 0  # Transition state
        is_synced = False  # Sync frame detected

        # Results stored in list
        raw_bits: List[int] = []
        decode_results: List[int] = []

        # Get sync
        for idx in range(len(raw_lux_diff_binary)):
            if not is_synced:
                if raw_lux_diff_binary[idx] == 0 and sync_started:
                    sync_interval += raw_lux_dt[idx]
                    continue
                elif raw_lux_diff_binary[idx] == 1:
                    sync_interval = 0
                    sync_started = True
                    continue
                elif raw_lux_diff_binary[idx] == -1:
                    if sync_interval > 2.8 * PERIOD:
                        is_synced = True
                        # First sync, init value_interval to 0, state to -1, raw_bits to [1] * 3
                        value_interval = 0
                        state = -1
                        raw_bits = [1] * 3
                    else:
                        sync_interval = 0
                        sync_started = False
                        continue

            if is_synced:
                if raw_lux_diff_binary[idx] == 0 or raw_lux_diff_binary[idx] == state:
                    # Increase value_interval
                    value_interval += raw_lux_dt[idx]
                else:
                    # Append to raw_bits on state change
                    if 1.5 * PERIOD > value_interval:
                        raw_bits += [state] * 1
                        value_interval = 0
                    elif 2.5 * PERIOD > value_interval >= 1.5 * PERIOD:
                        raw_bits += [state] * 2
                        value_interval = 0
                    else:
                        raw_bits.append(-1 if raw_bits[-1] == 1 else 1)
                        if len(raw_bits) >= 20:
                            result = self.decode_manchester_byte(raw_bits)
                            if result > 0:
                                decode_results.append(result)
                        is_synced = False
                        value_interval = 0
                    # Change state
                    state = raw_lux_diff_binary[idx]

        if len(decode_results) == 0:
            self.idx = -1
        else:
            self.idx = Counter(decode_results).most_common(1)[0][0]

        return self.idx


class TrackerUtils:
    def __init__(self):
        pass

    @classmethod
    def list_directory(cls, measurement_dir: str):
        sub_dirs = os.listdir(measurement_dir)
        sub_dirs = list(filter(lambda x: os.path.isdir(os.path.join(measurement_dir, x)), sub_dirs))
        res = {'measurement_dir': measurement_dir, 'sub_dirs': sub_dirs, 'data_path': {}}
        for sub_dir in res['sub_dirs']:
            meta = glob.glob(os.path.join(res['measurement_dir'], sub_dir, '*.csv'))[0]  # FORMAT RELATED
            mvdata = glob.glob(os.path.join(res['measurement_dir'], sub_dir, '*.mp4'))[0]  # FORMAT RELATED
            res['data_path'][sub_dir] = {'meta': meta, 'mvdata': mvdata}

        return res

    @classmethod
    def open_resources(cls, measurement_ctx):
        res = {}
        for camera_sn in measurement_ctx['sub_dirs']:
            cap = cv2.VideoCapture(measurement_ctx['data_path'][camera_sn]['mvdata'])
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                res[camera_sn] = {
                    'cap': cap,
                    'fps': fps,
                    'dt': 1 / fps,
                    'timestamp': pd.read_csv((measurement_ctx['data_path'][camera_sn]['meta'])).to_numpy()[:, 1:2],
                    'frame_count': frame_count
                    # FORMAT RELATED
                }

        return res

    @classmethod
    def filter_bandpass(cls,
                        data: np.ndarray,
                        band: Tuple[float, float] = (0.001, 8),
                        order: int = 7,
                        sample_freq: float = 100):
        res: np.array = np.copy(data)
        sos = signal.butter(order, tuple(map(lambda x: 2 * x / sample_freq, band)), 'bandpass', output='sos')
        for i in range(res.shape[-1]):
            res[..., i] = signal.sosfiltfilt(sos, data[..., i])

        # FIXME: Experimental
        # res -= np.linspace(res[0,:],np.array([0,0,0]), len(res))
        return res


class Tracker:
    @classmethod
    def analyse_frame(cls,
                      frame,
                      thresh,
                      erode_size,
                      erode_iterations,
                      blur_size=11,
                      curr_time=time.time(), frame_no=-1):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_frame = cv2.GaussianBlur(gray_frame, (blur_size, blur_size), 0)
        thresh, binary_frame = cv2.threshold(blur_frame, thresh, 255, cv2.THRESH_BINARY)
        binary_frame = cv2.erode(binary_frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size)),
                                 iterations=erode_iterations)
        binary_frame = cv2.dilate(binary_frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size)),
                                  iterations=erode_iterations)

        contours, hierarchy = cv2.findContours(binary_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours((contours, hierarchy))
        center_frame = origin_frame.copy()
        # 遍历轮廓集

        centers = []
        for c in cnts:
            M = cv2.moments(c)
            cX = int(M["m10"] / (1e-4 + M["m00"]))
            cY = int(M["m01"] / (1e-4 + M["m00"]))
            # http://edu.pointborn.com/article/2021/11/19/1709.html
            radius = math.sqrt(M['m00'])
            # 在图像上绘制轮廓及中心
            cv2.drawContours(center_frame, [c], -1, (0, 255, 0), 1)
            cv2.circle(center_frame, (cX, cY), int(radius), (255, 255, 255), 1)
            cv2.putText(center_frame, "center", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            centers.append({"x": cX, "y": cY, "r": radius, "t": curr_time, "frame_no": frame_no})

        res = {"timestamp": curr_time, "frame_no": frame_no, "centers": centers}
        return center_frame, res

    @classmethod
    def track(cls, analyse_results: List[Dict], track_minimal_distance: float):
        # print(analyse_results)
        # INPUT: analyse_results
        # OUTPUT: List[TrackSequence]
        finished_track_sequences = []
        active_track_sequences = list(map(lambda x: TrackSequence(x), analyse_results[0]["centers"]))
        new_track_sequences = []
        with tqdm.tqdm(range(len(analyse_results) - 1)) as pbar:
            for res in analyse_results[1:]:
                curr_centers = res["centers"]
                for center in curr_centers:
                    is_tracked = False
                    for track_seq in active_track_sequences:
                        if track_seq - center < track_minimal_distance:
                            track_seq.append(center)  # This will set track active
                            is_tracked = True
                            break
                    if not is_tracked:
                        new_track_sequences.append(TrackSequence(center))
                # Move non-active sequences to finished
                finished_track_sequences.extend([seq for seq in active_track_sequences if not seq.is_active])
                active_track_sequences = list(filter(lambda x: x.is_active, active_track_sequences))
                # Active sequences take step
                list(map(lambda x: x.step(), active_track_sequences))
                active_track_sequences.extend(new_track_sequences)
                new_track_sequences = []
                pbar.update()

        finished_track_sequences.extend(active_track_sequences)
        # Remove too short sequences
        finished_track_sequences = list(filter(lambda x: x.duration > 1.5, finished_track_sequences))
        print(f"Num of sequences: {len(finished_track_sequences)}")
        return finished_track_sequences


if __name__ == '__main__':


    MEASUREMENT_DIR = r"E:\Data\liyutong\record\highspeed\test_tutian-0b8e5cb4-2022-03-05_172423"

    # Hyper parameters

    BLUR_SIZE = 11  # Filter with cv2.GaussianBlur for better thresh segmentation
    THRESH = 40  # Brightness thresh
    ERODE_ITERATIONS: int = 0  # Erode intensity
    ERODE_SIZE: int = 2  # Structure size, 2 or 3 is OK
    TRACK_MINIMAL_DISTANCE = 32
    VISUALIZE = False

    # INPUT: MEASUREMENT_DIR
    # OUTPUT: analyse_results
    measurement_ctx = TrackerUtils.list_directory(MEASUREMENT_DIR)
    resources: Dict = TrackerUtils.open_resources(measurement_ctx)
    analyse_results: Dict[str, List[Dict]] = {}
    track_data: Dict[str, List[List]] = {}
    # [
    #  {
    #  "timestamp": float
    #  "centers": [
    #                {"x": float, # x coordinate
    #                 "y": float, # y coordinate
    #                 "r": float, # radius,
    #                 "t": float, # time,
    #                 "frame_no": int  # frame no
    #                },...
    #              ]
    #  },...
    # ]
    #
    for camera_sn in resources.keys():

        camera_resource = resources[camera_sn]
        analyse_results[camera_sn] = []
        cap = camera_resource['cap']  # Read video
        frames = []
        frame_num = 0

        with tqdm.tqdm(range(int(camera_resource['frame_count']))) as pbar:
            while cap.isOpened() and frame_num < len(camera_resource['timestamp']):
                ret, origin_frame = cap.read()
                if not ret:
                    cv2.destroyAllWindows()
                    break
                frames.append(origin_frame)
                center_frame, res = Tracker.analyse_frame(origin_frame, THRESH, ERODE_SIZE, ERODE_ITERATIONS, BLUR_SIZE,
                                                          float(camera_resource['timestamp'][frame_num]), frame_num)
                frame_num += 1
                analyse_results[camera_sn].append(res)
                pbar.update()

                if VISUALIZE:
                    cv2.imshow('Analyse', center_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break

        # print(analyse_results)
        # INPUT: analyse_results
        # OUTPUT: List[TrackSequence]
        finished_track_sequences = Tracker.track(analyse_results[camera_sn], TRACK_MINIMAL_DISTANCE)

        # INPUT: List of TrackSequences
        # OUTPUT: Index of imu
        [seq.process_track(frames) for seq in finished_track_sequences]

        track_data[camera_sn] = [[] for _ in range(frame_num)]
        for seq in finished_track_sequences:
            for center in seq.seq:
                x, y, r, t, no = center['x'], center['y'], int(center['r']), center['t'], center['frame_no']
                track_data[camera_sn][no].append({'x': x, 'y': y, 'r': r, 'idx': seq.idx, 'no': no, 'timestamp': t})

        if VISUALIZE:
            if not os.path.exists('example/out'):
                os.makedirs('example/out')
            for frame_no in range(frame_num):
                frame = frames[frame_no]
                for point in track_data[camera_sn][frame_no]:
                    # 在图像上绘制轮廓及中心
                    cv2.circle(frame, (point['x'], point['y']), int(point['r']), (255, 255, 255), 1)
                    cv2.putText(frame, str(point['idx']), (point['x'] - 20, point['y'] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow('demo_control', frame)
                cv2.imwrite(f'./out/{frame_num}.jpg', frame)
                frame_num += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        cv2.destroyAllWindows()

    # Collect date for visualisation
    track_data_json_output = os.path.join(MEASUREMENT_DIR, f'track_data.json')
    with open(track_data_json_output, 'w') as f:
        json.dump(track_data, f)

    print('finish')
