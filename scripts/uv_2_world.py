import json
import os.path as osp

import numpy as np

MEASUREMENT_DIR = r"E:\Data\liyutong\record\highspeed\test_tutian-0b8e5cb4-2022-03-05_172423"

CAMS = [
    '041190220306',
    '041131220295',
    '041131220297',
    '041190220307',
]

CAM_NAMES = [
    'h0',
    'h1',
    'h2',
    'h3',
]

calibration_result = json.load(open(osp.join(MEASUREMENT_DIR, 'calibration.json')))
track_result = json.load(open(osp.join(MEASUREMENT_DIR, 'track_data.json')))
uv_result = dict()  # cam1: {ts1: {}, ts2: {}, ts3: {}, ...}, cam2: {}, cam3: {}, ...

for cam in CAMS:
    result = track_result[cam]
    result_to_update = dict()

    for frame in result:
        assert (len(frame)) <= 1
        if len(frame) == 0:
            continue
        # else
        result_to_update[frame[0]['timestamp']] = frame[0]

    uv_result[cam] = result_to_update

# align
cam_to_ts = {cam: (np.array((0,), dtype=np.float64)
                   if len(uv_result[cam]) == 0
                   else np.array(list(uv_result[cam].keys()), dtype=np.float64))
             for cam in CAMS}
all_ts = set()
for cam in CAMS:
    all_ts.update(list(uv_result[cam].keys()))
all_ts = list(all_ts)
all_ts.sort()

TS_THRESHOLD = 2.0 / 1000
aligned_ts = dict()
for ts in all_ts:
    cur_ts_2_cam_ts = {cam: None for cam in CAMS}
    for cam in CAMS:
        delta_tss = np.abs(cam_to_ts[cam] - ts)
        min_delta_idx = np.argmin(delta_tss)
        min_delta_value = delta_tss[min_delta_idx]
        if min_delta_value < TS_THRESHOLD:
            cur_ts_2_cam_ts[cam] = cam_to_ts[cam][min_delta_idx]
    aligned_ts[ts] = cur_ts_2_cam_ts

# uv mapping
localization_result = []
for ts, cam_ts_dict in aligned_ts.items():
    Ni = []
    ai = []
    h_cams_p = []
    for camera, cam_name in zip(CAMS, CAM_NAMES):
        if cam_ts_dict[camera] is None:
            continue
        # else
        this_cam_ts = cam_ts_dict[camera]
        cur_uv_result = uv_result[camera][this_cam_ts]
        x, y, led_idx = cur_uv_result['x'], 1200 - cur_uv_result['y'], cur_uv_result['idx']
        fx = float(calibration_result['cameras'][cam_name]['K'][0][0])
        fy = float(calibration_result['cameras'][cam_name]['K'][1][1])
        cx = float(calibration_result['cameras'][cam_name]['K'][0][2])
        cy = float(calibration_result['cameras'][cam_name]['K'][1][2])

        camera_extrinsic = np.zeros((4, 4), dtype=np.float32)
        R = calibration_result['camera_poses'][cam_name + ('_to_h0' if cam_name != 'h0' else '')]['R']
        T = calibration_result['camera_poses'][cam_name + ('_to_h0' if cam_name != 'h0' else '')]['T']
        camera_extrinsic[:3, :3] = np.array(R)
        camera_extrinsic[:3, 3] = np.array(T)
        camera_extrinsic[3, 3] = 1
        # ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6614647

        a = np.array((0, 0, 0, 1), dtype=float).reshape((4, 1))
        a = np.matmul(np.linalg.inv(camera_extrinsic), a)
        a = a[:3]
        ai.append(a)

        v = np.array(((x - cx) / fx, (y - cy) / fy, 1), ).reshape((3, 1))
        v = v / np.linalg.norm(v)
        _ = np.ones((4,), dtype=float).reshape((4, 1))
        _[:3] = v
        v = _
        v = np.matmul(np.linalg.inv(camera_extrinsic), v)[:3] - a

        N = np.identity(3, dtype=float) - v * v.T
        Ni.append(N)

    if len(Ni) > 1:
        p = np.matmul(np.linalg.inv(sum(N for N in Ni)), (sum(np.matmul(N, a) for N, a in zip(Ni, ai))))

        cur_info_dict = dict(
            timestamp=ts,
            position=p.flatten().tolist(),
            num_cam=len(Ni),
        )
        localization_result.append(cur_info_dict)

print(len(localization_result))

with open(osp.join(MEASUREMENT_DIR, 'localization_data.json'), 'w') as f:
    json.dump(localization_result, f)
