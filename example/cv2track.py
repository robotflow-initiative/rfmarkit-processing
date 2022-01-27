import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import math
from typing import List, Dict
import tqdm
from collections import Counter


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
    def analyse_frame(cls,
                      frame,
                      thresh,
                      erode_size,
                      erode_iterations,
                      blur_size = 11,
                      curr_time=time.time(), frame_no=-1):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_frame = cv2.GaussianBlur(gray_frame, (blur_size, blur_size), 0)
        thresh, binary_frame = cv2.threshold(blur_frame, thresh, 255, cv2.THRESH_BINARY)
        binary_frame = cv2.erode(binary_frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size)),
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

    def process_track(self):
        DIFF_THRESH = 2000  # Difference in luminance between light/dark frames
        PERIOD = 0.05

        raw_lux = np.zeros(shape=(len(self),), dtype=float)
        raw_lux_dt = np.zeros(shape=(len(self),), dtype=float)
        for idx, center in enumerate(self.seq):
            x, y, r, t, frame_no = center['x'], center['y'], int(center['r']), center['t'], center['frame_no']
            raw_lux[idx] = np.sum(frames[frame_no][y - r:y + r, x - r:x + r])  # Get luminance
            raw_lux_dt[idx] = t  # Temporarily save t，this is not dt

        raw_lux_dt[1:] = raw_lux_dt[1:] - raw_lux_dt[:-1]  # Finally get frame deltas
        raw_lux_dt[0] = 0  # Finally get frame deltas
        # Usually the delta is small
        frame_delta_avg, frame_delta_std = raw_lux_dt.mean(), raw_lux_dt.std()
        print(f"frame delta avg={frame_delta_avg}, std={frame_delta_std}")

        # Get frame luminance difference
        raw_lux_diff = np.zeros_like(raw_lux)
        raw_lux_diff[1:] = raw_lux[1:] - raw_lux[:-1]
        # Convert to binary using threshold
        raw_lux_diff_binary = (raw_lux_diff > DIFF_THRESH).astype(np.int64) - (raw_lux_diff < - DIFF_THRESH).astype(
            np.int64)
        # Visualisation
        plt.plot(np.linspace(0, self.duration, len(self)), raw_lux_diff_binary, linewidth=1)
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


if __name__ == '__main__':
    # Hyper parameters
    FILE = "./IMG_0504.MOV"
    BLUR_SIZE = 11  # Filter with cv2.GaussianBlur for better thresh segmentation
    THRESH = 50  # Brightness thresh
    ERODE_ITERATIONS: int = 1  # Erode intensity
    ERODE_SIZE: int = 2  # Structure size, 2 or 3 is OK
    TRACK_MINIMAL_DISTANCE = 16

    # Analyse result
    analyse_results: List[Dict] = []
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

    # INPUT: FILE
    # OUTPUT: analyse_results
    cap = cv2.VideoCapture(FILE)  # Read video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"fps={fps}")
    dt = 1 / fps
    curr_time = 0
    frames = []
    frame_no = 0
    while cap.isOpened():
        ret, origin_frame = cap.read()
        if not ret:
            cv2.destroyAllWindows()
            break
        gray_frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2GRAY)
        blur_frame = cv2.GaussianBlur(gray_frame, (BLUR_SIZE, BLUR_SIZE), 0)
        frames.append(blur_frame)
        center_frame, res = TrackSequence.analyse_frame(origin_frame, THRESH, ERODE_SIZE, ERODE_ITERATIONS, curr_time, frame_no)
        curr_time += dt
        frame_no += 1
        analyse_results.append(res)

        cv2.imshow('demo_control', center_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # print(analyse_results)
    # INPUT: analyse_results
    # OUTPUT: List[TrackSequence]
    finished_track_sequences = TrackSequence.track(analyse_results, TRACK_MINIMAL_DISTANCE)

    # INPUT: List of TrackSequences
    # OUTPUT: Index of imu
    [seq.process_track() for seq in finished_track_sequences]

    # Collect date for visualisation
    data = [[] for _ in range(len(frames))]
    for seq in finished_track_sequences:
        for center in seq.seq:
            x, y, r, t, no = center['x'], center['y'], int(center['r']), center['t'], center['frame_no']
            data[no].append({'x': x, 'y': y, 'r': r, 'idx': seq.idx, 'no': no})

    frame_no = 0
    os.makedirs('./out')
    for gray_frame in frames:
        for point in data[frame_no]:
            # 在图像上绘制轮廓及中心
            cv2.circle(gray_frame, (point['x'], point['y']), int(point['r']), (255, 255, 255), 1)
            cv2.putText(gray_frame, str(point['idx']), (point['x'] - 20, point['y'] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('demo_control', gray_frame)
        cv2.imwrite(f'./out/{frame_no}.jpg', gray_frame)
        frame_no += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    print('finish')
