import datetime
import json
import os
import os.path as osp
from typing import List, Dict, Any

import cv2
import numpy as np
import tqdm
from norfair import Tracker, draw_tracked_objects, Detection
from realsense_recorder.io import DirectoryReader
from rich.console import Console

from markit_processing.datamodels import RecordingModel
from markit_processing.detector import OpenCVDetector

# Hyper parameters
CONFIG_DETECTOR_BLUR_SIZE = 3  # Filter with cv2.GaussianBlur for better thresh segmentation
CONFIG_DETECTOR_THRESH = 36  # Brightness thresh
CONFIG_DETECTOR_N_ERODE_ITERATIONS: int = 0  # Erode intensity
CONFIG_DETECTOR_N_DILATE_ITERATIONS: int = 2
CONFIG_DETECTOR_ERODE_SIZE: int = 1  # Structure size, 2 or 3 is OK
CONFIG_DETECTOR_DEBUG = False
CONFIG_FG_DETECTION_ENABLED = False
CONFIG_FG_HISTORY = 5
CONFIG_FG_VAR_THRESHOLD = 16
CONFIG_READER_N_PRELOAD = 4
CONFIG_TRACKER_MINIMAL_DISTANCE = 32
CONFIG_CAMERA_MASK = ["r22", "r69"]


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def get_wrapped_detection(raw_detections):
    return [Detection(detection[:2]) for detection in raw_detections]


def update_tracking_result(result: Dict[str, Dict[str, List[Dict[str, Any]]]],
                           camera_friendly_name: str,
                           sequence_idx: int,
                           tracked_objects: List[Any],
                           meta: Dict[str, Any]):
    for target in tracked_objects:
        if target.id not in result[camera_friendly_name].keys():
            result[camera_friendly_name][target.id] = []

        result[camera_friendly_name][target.id].append({
            "sequence_idx": sequence_idx,
            "meta": meta,
            "position": target.estimate.tolist()
        })
    pass


def run_once(recording_path: str):
    # Norfair
    recording = RecordingModel(recording_path)
    recording.load()

    tracking_result = {
        camera_friendly_name: {} for camera_friendly_name in CONFIG_CAMERA_MASK
    }
    for name in filter(lambda x: x in recording.realsense_stream.recordings.keys(), CONFIG_CAMERA_MASK):
        detector = OpenCVDetector(thresh=CONFIG_DETECTOR_THRESH,
                                  erode_size=CONFIG_DETECTOR_ERODE_SIZE,
                                  n_erode_iterations=CONFIG_DETECTOR_N_ERODE_ITERATIONS,
                                  n_dilate_iterations=CONFIG_DETECTOR_N_DILATE_ITERATIONS,
                                  blur_size=CONFIG_DETECTOR_BLUR_SIZE,
                                  fg_detection_enabled=CONFIG_FG_DETECTION_ENABLED,
                                  fg_history=CONFIG_FG_HISTORY,
                                  fg_var_threshold=CONFIG_FG_VAR_THRESHOLD)
        tracker = Tracker(distance_function=euclidean_distance,
                          distance_threshold=20)

        video: DirectoryReader = recording.realsense_stream.recordings[name].color

        with tqdm.tqdm(total=len(video)) as pbar:
            while not video.eof:
                frame, meta, sequence_idx = video.next()
                rendered_frame, raw_detections, _ = detector.process_frame(frame, CONFIG_DETECTOR_DEBUG)
                wrapped_detections = get_wrapped_detection(raw_detections)
                tracked_objects = tracker.update(detections=wrapped_detections)
                draw_tracked_objects(frame, tracked_objects)
                update_tracking_result(tracking_result, name, sequence_idx, tracked_objects, meta)
                if CONFIG_DETECTOR_DEBUG:
                    cv2.imshow('Analyse', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
                pbar.update(1)

    with open(osp.join(recording_path, "realsense", "led_tracking_result.json"), "w") as f:
        json.dump({
            "meta": {
                "recording": osp.basename(recording_path),
                "cameras": list(tracking_result.keys()),
            },
            "result": tracking_result
        }, f, indent=4)


def main():
    with open('../logs/' + "run_norfair_detection_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log", 'w') as f:
        console = Console(file=f)

        # FIXME: Change this to your own path
        DATA_DIR = r"D:\pre-release"
        IMMOBILE_DIR = osp.join(DATA_DIR, "data", "immobile")
        PORTABLE_DIR = osp.join(DATA_DIR, "data", "portable")
        TARGETS = list(filter(lambda x: osp.isdir(x), [osp.join(IMMOBILE_DIR, x) for x in os.listdir(IMMOBILE_DIR)] + [osp.join(PORTABLE_DIR, x) for x in os.listdir(PORTABLE_DIR)]))
        print(TARGETS)
        with tqdm.tqdm(total=len(TARGETS)) as pbar:
            for target in TARGETS:
                try:
                    run_once(target)
                    console.log(f"Finished {target}")
                except Exception as _:
                    console.log(f"Failed {target}")
                pbar.update(1)


if __name__ == '__main__':
    # run_once(r"D:\pre-release\data\immobile\eyeglasses-049-1")
    main()
