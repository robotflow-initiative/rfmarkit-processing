import time

import cv2
import numpy as np
import tqdm

from markit_processing.detector import OpenCVDetector
from realsense_recorder.io import get_directory_reader, DirectoryReader
from markit_processing.datamodels import RecordingModel


def _detection_job(d: OpenCVDetector, reader: DirectoryReader):
    with tqdm.tqdm(range(len(reader))) as pbar:
        while not reader.eof:
            frame, meta = reader.next()
            if frame is None:
                cv2.destroyAllWindows()
                break
            rendered_frame, res, _ = d.process_frame(frame, CONFIG_DETECTOR_DEBUG)
            pbar.update()

            if CONFIG_DETECTOR_DEBUG:
                cv2.imshow('Analyse', rendered_frame)
                time.sleep(0.2)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break


if __name__ == '__main__':
    CONFIG_RECORDING_DIR = r"D:\pre-release\data\immobile\bottle-014-1"

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
    CONFIG_TRACKER_MINIMAL_DISTANCE = 32
    CONFIG_CAMERA_MASK = ["011422071122"]

    # INPUT: MEASUREMENT_DIR
    # OUTPUT: analyse_results

    recording = RecordingModel(CONFIG_RECORDING_DIR)
    recording.load()
    Det = OpenCVDetector(thresh=CONFIG_DETECTOR_THRESH,
                         erode_size=CONFIG_DETECTOR_ERODE_SIZE,
                         n_erode_iterations=CONFIG_DETECTOR_N_ERODE_ITERATIONS,
                         n_dilate_iterations=CONFIG_DETECTOR_N_DILATE_ITERATIONS,
                         blur_size=CONFIG_DETECTOR_BLUR_SIZE,
                         fg_detection_enabled=CONFIG_FG_DETECTION_ENABLED,
                         fg_history=CONFIG_FG_HISTORY,
                         fg_var_threshold=CONFIG_FG_VAR_THRESHOLD)
    _detection_job(Det, recording.realsense_stream.recordings['r22'].color)

    print('finish')
