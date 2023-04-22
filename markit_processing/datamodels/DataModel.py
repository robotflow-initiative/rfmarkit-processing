import dataclasses
import glob
import json
import logging
import os.path as osp
import pickle
from typing import Optional

import cv2
import numpy as np
import tqdm
from realsense_recorder.io import DirectoryReader, get_directory_reader

logger = logging.getLogger('markit_processing.datamodels.DataModel')

def _read_function(path: str) -> np.ndarray:
    """
    Read anything, from a path
    :param path:
    :return: numpy.ndarray, RGB frame
    """
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _parse_function(label: str):
    """
    Parse meta data from label as string
    :param label:
    :return:
    """
    basename = osp.basename(label)
    meta = osp.splitext(basename)[0].split('_')
    return {
        "frame_idx": meta[0],
        "ts": meta[1],
        "sys_ts": meta[2],
        "basename": basename
    }


@dataclasses.dataclass()
class PreReleaseRealsenseStreamModelPerCamera:
    path_to_recording: str
    num_preload: int = 8
    path_to_color: str = ''
    path_to_depth: str = ''
    color: DirectoryReader = dataclasses.field(init=False)
    depth: DirectoryReader = dataclasses.field(init=False)

    def __post_init__(self):
        self.path_to_color = osp.join(self.path_to_recording, 'color')
        self.path_to_depth = osp.join(self.path_to_recording, 'depth')

    def load(self, selected_frames: Optional[dict] = None):
        self.color = get_directory_reader(self.path_to_color, 'color_jpeg', self.num_preload, read_function=_read_function, parse_function=_parse_function)
        self.depth = get_directory_reader(self.path_to_color, 'depth_npz', self.num_preload, read_function=_read_function, parse_function=_parse_function)
        if selected_frames is not None:
            self.color.reload(list(map(lambda x: osp.join(self.path_to_color, x), selected_frames['color'])), sort=True)
            self.depth.reload(list(map(lambda x: osp.join(self.path_to_color, x), selected_frames['color'])), sort=True)


@dataclasses.dataclass()
class RealsenseStreamModel:
    path_to_stream: str
    path_to_recordings: dict = dataclasses.field(default_factory=dict)
    path_to_metadata: str = ''
    path_to_selected_frames: str = ''
    path_to_config: str = ''
    recordings: dict = dataclasses.field(init=False)
    metadata: dict = dataclasses.field(init=False)
    config: dict = dataclasses.field(init=False)
    selected_frames: Optional[dict] = dataclasses.field(init=False)
    led_tracking_result: Optional[dict] = dataclasses.field(init=False)
    marker_detection_result: Optional[dict] = dataclasses.field(init=False)
    camera_friendly_names = property(lambda self: list(self.path_to_recordings.keys()))
    path_to_cameras = property(lambda self: [osp.join(self.path_to_stream, name) for name in self.camera_friendly_names])

    def __post_init__(self):
        path_list = list(filter(lambda x: osp.isdir(x), glob.glob(osp.join(self.path_to_stream, '*'))))
        camera_friendly_names = list(map(lambda x: osp.basename(x), path_list))
        self.path_to_recordings = {device_id: path for device_id, path in zip(camera_friendly_names, path_list)}
        self.recordings = {device_id: PreReleaseRealsenseStreamModelPerCamera(path) for device_id, path in zip(camera_friendly_names, path_list)}
        self.path_to_metadata = osp.join(self.path_to_stream, 'metadata_all.json')
        self.path_to_selected_frames = osp.join(self.path_to_stream, 'selected_frames.json')
        self.path_to_led_tracking_result = osp.join(self.path_to_stream, 'led_tracking_result.json')
        self.path_to_marker_detection_result = osp.join(self.path_to_stream, 'marker_detection_result.json')

        self.path_to_config = osp.join(self.path_to_stream, 'realsense_config.json')

    def load(self):
        self.metadata = json.load(open(self.path_to_metadata))
        self.config = json.load(open(self.path_to_config))
        if osp.exists(self.path_to_selected_frames):
            try:
                self.selected_frames = json.load(open(self.path_to_selected_frames))
            except json.decoder.JSONDecodeError:
                self.selected_frames = None
        else:
            self.selected_frames = None

        if osp.exists(self.path_to_led_tracking_result):
            try:
                self.led_tracking_result = json.load(open(self.path_to_led_tracking_result))
            except json.decoder.JSONDecodeError:
                self.led_tracking_result = None
        else:
            self.led_tracking_result = None

        if osp.exists(self.path_to_marker_detection_result):
            try:
                self.marker_detection_result = json.load(open(self.path_to_led_tracking_result))
            except json.decoder.JSONDecodeError:
                self.marker_detection_result = None
        else:
            self.marker_detection_result = None

        for camera_friendly_name, item in self.recordings.items():
            if self.selected_frames is not None:
                item.load(self.selected_frames['filenames'][camera_friendly_name])
            else:
                item.load()


@dataclasses.dataclass()
class IMUStreamModel:
    path_to_stream: str
    path_to_recordings: dict = dataclasses.field(default_factory=dict)
    recordings: dict = dataclasses.field(init=False)
    synced_recording: Optional[dict] = dataclasses.field(init=False)
    synced_recording_metadata: Optional[dict] = dataclasses.field(init=False)

    def __post_init__(self):
        path_list = glob.glob(osp.join(self.path_to_stream, '*.npz'))
        self.synced_recording_path = osp.join(self.path_to_stream, "imu_all.pkl")
        self.synced_recording_metadata_path = osp.join(self.path_to_stream, "imu.json")
        device_id_list = list(map(lambda x: osp.splitext(osp.basename(x))[0].split('_')[1], path_list))
        self.path_to_recordings = {device_id: path for device_id, path in zip(device_id_list, path_list)}
        pass

    def load(self):
        self.recordings = {device_id: np.load(path) for device_id, path in self.path_to_recordings.items()}
        if osp.exists(self.synced_recording_path):
            try:
                self.synced_recording = pickle.load(open(self.synced_recording_path, 'rb'))
            except Exception as e:
                print(e)
                self.synced_recording = None
        else:
            self.synced_recording = None

        if osp.exists(self.synced_recording_metadata_path):
            try:
                self.synced_recording_metadata = json.load(open(self.synced_recording_metadata_path))
            except Exception as e:
                print(e)
                self.synced_recording_metadata = None
        else:
            self.synced_recording_metadata = None


class RecordingModel:
    def __init__(self, path_to_recording):
        self.path_to_recording = path_to_recording
        path_to_streams = {
            'imu': osp.join(self.path_to_recording, 'imu'),
            'realsense': osp.join(self.path_to_recording, 'realsense'),
        }
        self.imu_stream = IMUStreamModel(path_to_streams['imu'])
        self.realsense_stream = RealsenseStreamModel(path_to_streams['realsense'])

    def load(self):
        self.imu_stream.load()
        self.realsense_stream.load()


if __name__ == '__main__':
    x = RecordingModel(r"D:\pre-release\data\immobile\bottle-014-1")
    x.load()
    print(x)
    readers = [x.color for x in x.realsense_stream.recordings.values()]
    with tqdm.tqdm(total=len(readers) * len(readers[0])) as pbar:
        while not any([reader.eof for reader in readers]):
            [reader.next() for reader in readers]
            pbar.update(len(readers))

    print(all([reader.eof for reader in readers]))
