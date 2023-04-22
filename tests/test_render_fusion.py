import json
import json
import os.path as osp
from typing import List, Dict, Any

import cv2
import numpy as np
import tqdm
from norfair import Detection
from realsense_recorder.io import get_directory_reader

from markit_processing.detector import OpenCVDetector

# Hyper parameters
CONFIG_DETECTOR_BLUR_SIZE = 3  # Filter with cv2.GaussianBlur for better thresh segmentation
CONFIG_DETECTOR_THRESH = 36  # Brightness thresh
CONFIG_DETECTOR_N_ERODE_ITERATIONS: int = 0  # Erode intensity
CONFIG_DETECTOR_N_DILATE_ITERATIONS: int = 2
CONFIG_DETECTOR_ERODE_SIZE: int = 1  # Structure size, 2 or 3 is OK
CONFIG_DETECTOR_DEBUG = True
CONFIG_FG_DETECTION_ENABLED = False
CONFIG_FG_HISTORY = 5
CONFIG_FG_VAR_THRESHOLD = 16
CONFIG_READER_N_PRELOAD = 4
CONFIG_TRACKER_MINIMAL_DISTANCE = 64
CONFIG_TRACKING_CAMERA = "r69"
CONFIG_RENDER_CAMERA = "r85"

r69_intrinsics =  {
      "model": "standard",
      "image_size": [
        1280,
        720
      ],
      "K": [
        [
          987.6411949273044,
          0.0,
          655.467168234867
        ],
        [
          0.0,
          1042.3912207281646,
          415.45775140330306
        ],
        [
          0.0,
          0.0,
          1.0
        ]
      ],
      "dist": [
        [
          0.08786137449273068,
          0.44987337435752317,
          0.004906562891000099,
          -0.00317908351457484,
          -2.8321869086353124
        ]
      ]
    },

r85_intrinsics = {
    "model": "standard",
    "image_size": [
        1280,
        720
    ],
    "K": [
        [
            1157.4214609018072,
            0.0,
            652.2173166388815
        ],
        [
            0.0,
            1271.4273927568747,
            396.2071016107469
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ],
    "dist": [
        [
            0.1927826544288516,
            -0.34972530095573834,
            0.011612480526787846,
            -0.00393533140166019,
            -2.9216752723525734
        ]
    ]
}

r69_to_r08 = {
      "R": [
        [
          0.13282964829377214,
          0.8271809337520729,
          -0.5460109773358056
        ],
        [
          -0.853745626695679,
          0.37532282676508516,
          0.3609032842841255
        ],
        [
          0.5034626991467742,
          0.41821582770874993,
          0.7560561037527109
        ]
      ],
      "T": [
        0.7882414302179732,
        -0.6522944669713474,
        0.2434025354770135
      ]
    },

r85_to_r08 = {
      "R": [
        [
          0.14870220218370309,
          0.828074371465997,
          -0.5405372238679881
        ],
        [
          -0.7935719239353951,
          0.4260703498140629,
          0.4344049476592063
        ],
        [
          0.5900264880555727,
          0.36435819234719646,
          0.720494171428396
        ]
      ],
      "T": [
        0.7601228932740565,
        -0.5973250348424212,
        0.4241104677493674
      ]
    }
def multical_dict_to_rot_matrix(x):
    return np.concatenate([np.concatenate([np.array(x["R"]), np.expand_dims(np.array(x["T"]),1).T], axis=1), np.array([[0,0,0,1]])], axis=0)

r85_to_r69 =  multical_dict_to_rot_matrix(r85_to_r08) @ np.inv(multical_dict_to_rot_matrix(r69_to_r08))
r69_to_r85 =   multical_dict_to_rot_matrix(r69_to_r08) @ np.inv(multical_dict_to_rot_matrix(r85_to_r08))

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def get_wrapped_detection(raw_detections):
    return [Detection(detection[:2]) for detection in raw_detections]


def update_tracking_result(result: Dict[str, List[Dict[str, Any]]],
                           sequence_idx: int,
                           tracked_objects: List[Any],
                           meta: Dict[str, Any]):
    for target in tracked_objects:
        if target.id not in result.keys():
            result[target.id] = []

        result[target.id].append({
            "sequence_idx": sequence_idx,
            "meta": meta,
            "position": target.estimate.tolist()
        })
    pass


def update_detecting_result(result: Dict[int, Any],
                            sequence_idx: int,
                            detection: np.ndarray,
                            meta: Dict[str, Any]):
    result[sequence_idx] = {
        "sequence_idx": sequence_idx,
        "meta": meta,
        "position": detection.tolist()
    }
    pass


def track_once(recording_path):
    tracking_result = {}

    detector = OpenCVDetector(thresh=CONFIG_DETECTOR_THRESH,
                              erode_size=CONFIG_DETECTOR_ERODE_SIZE,
                              n_erode_iterations=CONFIG_DETECTOR_N_ERODE_ITERATIONS,
                              n_dilate_iterations=CONFIG_DETECTOR_N_DILATE_ITERATIONS,
                              blur_size=CONFIG_DETECTOR_BLUR_SIZE,
                              fg_detection_enabled=CONFIG_FG_DETECTION_ENABLED,
                              fg_history=CONFIG_FG_HISTORY,
                              fg_var_threshold=CONFIG_FG_VAR_THRESHOLD)
    # tracker = Tracker(distance_function=euclidean_distance,
    #                   distance_threshold=128,
    #                   reid_distance_threshold=128,)

    video = get_directory_reader(osp.join(recording_path, 'realsense', CONFIG_TRACKING_CAMERA, "color"), "color_jpeg")

    with tqdm.tqdm(len(video)) as pbar:
        while not video.eof:
            frame, meta, sequence_idx = video.next()
            rendered_frame, raw_detections, _ = detector.process_frame(frame, CONFIG_DETECTOR_DEBUG)
            # wrapped_detections = get_wrapped_detection(raw_detections)
            # tracked_objects = tracker.update(detections=wrapped_detections)
            # draw_tracked_objects(frame, tracked_objects)
            update_detecting_result(tracking_result, sequence_idx, raw_detections, meta)
            if CONFIG_DETECTOR_DEBUG:
                cv2.imshow('Analyse', rendered_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            pbar.update(1)

    with open(osp.join(recording_path, "realsense", "led_tracking_result.json"), "w") as f:
        json.dump({
            "meta": {
                "recording": osp.basename(recording_path),
                "parts": list(tracking_result.keys()),
            },
            "result": tracking_result
        }, f, indent=4)


def collapse_id(tracking_result):
    res = 0 - np.ones(shape=(len(tracking_result) + 1, 2))

    for _, r in tracking_result.items():
        if len(r['position']) >= 1:
            res[r['sequence_idx']] = r['position'][0][:2]

    return res


def render_tracked_result(recording_path):
    with open(osp.join(recording_path, "realsense", "led_tracking_result.json")) as f:
        result = json.load(f)['result']
    result_collapsed = collapse_id(result)

    tracking_video = get_directory_reader(osp.join(recording_path, 'realsense', CONFIG_TRACKING_CAMERA, "color"), "color_jpeg")
    tracking_depth = get_directory_reader(osp.join(recording_path, 'realsense', CONFIG_TRACKING_CAMERA, "depth"), "depth_npz")
    render_video = get_directory_reader(osp.join(recording_path, 'realsense', CONFIG_RENDER_CAMERA, "color"), "color_jpeg")
    render_depth = get_directory_reader(osp.join(recording_path, 'realsense', CONFIG_RENDER_CAMERA, "depth"), "depth_npz")
    assert len(tracking_video) == len(tracking_depth) == len(render_video) == len(render_depth)

    while not tracking_video.eof:
        tracking_video_frame, tracking_video_meta, tracking_video_sequence_idx = tracking_video.next()
        tracking_depth_frame, tracking_depth_meta, tracking_depth_sequence_idx = tracking_depth.next()
        render_video_frame, render_video_meta, render_video_sequence_idx = render_video.next()
        render_depth_frame, render_depth_meta, render_depth_sequence_idx = render_depth.next()


        # cv2.circle(frame, (int(result_collapsed[sequence_idx][0]), int(result_collapsed[sequence_idx][1])), int(5), (255, 255, 255), 1)
        # cv2.imshow("frame", frame)
        # key = cv2.waitKey(10)
        #
        # if key == 27:  # 按esc键退出
        #     print('esc break...')
        #     cv2.destroyAllWindows()
        #     break


# track_once(r"C:\Users\liyutong\Downloads\video\demo")
render_tracked_result(r"C:\Users\liyutong\Downloads\video\demo")
