import datetime
import json
import os
import os.path as osp
from typing import List, Dict, Any, Optional

import numpy as np
import tqdm
from rich.console import Console

console: Optional[Console] = None

from internal.datamodels.PreReleaseDataModel import PreReleaseRecordingModel

METHOD: Optional[str] = None


def _get_file_name_from_metadata(metadata: List[Dict[str, Any]], idx, ext: str = 'jpeg'):
    return f"{idx + 1}_{metadata[idx]['ts']}_{metadata[idx]['sys_ts']}.{ext}"


def _get_camera_friendly_name(device_id: str):
    return 'r' + device_id[-2:]


def _get_master_camera(cameras: List[str], all_metadata: Dict[str, Any]):
    start_time = [all_metadata['metadata'][cam][0][METHOD] for cam in cameras]
    return cameras[start_time.index(max(start_time))]


def _query_filenames(selected_frames: Dict[str, List[int]], all_metadata: Dict[str, Any], cameras: List[str]) -> Dict[str, Dict[str, List[str]]]:
    return {
        cam: {
            'color': [_get_file_name_from_metadata(all_metadata['metadata'][cam], idx, ext="jpeg") for idx in selected_frames[cam]],
            'depth': [_get_file_name_from_metadata(all_metadata['metadata'][cam], idx, ext="npz") for idx in selected_frames[cam]]
        } for cam in cameras
    }


def _validate_filename_list(base_dir: str, filenames: Dict[str, Dict[str, List[str]]]):
    for cam in filenames.keys():
        for cam_stream in filenames[cam]:
            ls_result = os.listdir(osp.join(base_dir, 'realsense', cam, cam_stream))
            if not all([name in ls_result for name in filenames[cam][cam_stream]]):
                return False
    return True


def process_recording(path_to_recording: str, time_delta_threshold_ms: int = 33):
    global console
    recording = PreReleaseRecordingModel(path_to_recording)
    recording.load()
    all_metadata = recording.realsense_stream.metadata
    cameras: List[str] = list(map(lambda x: _get_camera_friendly_name(x), all_metadata['camera_sn']))
    master_camera = _get_master_camera(cameras, all_metadata)

    selected_frames = {
        cam: [] for cam in cameras
    }
    last_frame_counter = -1
    curr_frame_idx = 0
    slave_frame_idx = {
        cam: 0 for cam in cameras if cam is not master_camera
    }

    master_frame_counters = np.array([x['frame_counter'] for x in all_metadata['metadata'][master_camera]])
    master_frame_timestamps = np.array([x[METHOD] for x in all_metadata['metadata'][master_camera]])
    slave_frame_timestamps = {
        cam: np.array([x[METHOD] for x in all_metadata['metadata'][cam]]) for cam in cameras if cam != master_camera
    }
    num_dropped_frames = 0
    for curr_frame_idx in range(len(master_frame_counters)):
        curr_frame_counter = master_frame_counters[curr_frame_idx]  # read frame counter from master camera

        if curr_frame_counter > last_frame_counter:  # if it is a new frame
            last_frame_counter = curr_frame_counter  # update last frame counter

            # check if slave cameras have matched frame
            is_valid_frame = True
            for cam in filter(lambda x: x != master_camera, cameras):
                time_delta = np.abs(slave_frame_timestamps[cam] - master_frame_timestamps[curr_frame_idx])
                minimum_time_delta = np.min(time_delta)
                if minimum_time_delta > time_delta_threshold_ms:
                    is_valid_frame = False
                    if np.argmin(time_delta) < len(slave_frame_timestamps[cam]) - 1:  # not the last frame
                        num_dropped_frames += 1
                    break
                else:
                    slave_frame_idx[cam] = np.argmin(time_delta)
            if not is_valid_frame:
                continue
            else:
                [selected_frames[cam].append(slave_frame_idx[cam]) for cam in slave_frame_idx.keys()]
                selected_frames[master_camera].append(curr_frame_idx)
    filenames = _query_filenames(selected_frames, all_metadata, cameras)
    # Assert all frames have equal number of frames
    assert min([len(x) for x in filenames[master_camera].values()]) == min([len(x) for x in filenames[master_camera].values()])
    is_validate_recording = _validate_filename_list(path_to_recording, filenames)

    console.print("----------------------------------------")
    console.print("Processing recording: ", path_to_recording)
    console.print("Validation result: ", is_validate_recording)
    console.print("Number of Selected frames: ", len(selected_frames[master_camera]))
    console.print("Number of Dropped frames: ", num_dropped_frames)
    console.print("----------------------------------------\n")
    print("----------------------------------------")
    print("Processing recording: ", path_to_recording)
    print("Validation result: ", is_validate_recording)
    print("Number of Selected frames: ", len(selected_frames[master_camera]))
    print("Number of Dropped frames: ", num_dropped_frames)
    print("----------------------------------------\n")

    with open(osp.join(path_to_recording, "realsense", "selected_frames.json"), 'w') as f:
        json.dump({
            "meta": {
                "master_camera": master_camera,
                "slave_cameras": list(filter(lambda x: x != master_camera, cameras)),
                "num_selected_frames": len(selected_frames[master_camera]),
                "num_dropped_frames": num_dropped_frames,
            },
            "filenames": filenames
        }, f, indent=4)


def main():
    global console, METHOD
    with open('./' + "run_video_alignment_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log", 'w') as f:
        console = Console(file=f)

        # FIXME: Change this to your own path
        METHOD = 'ts'  # or 'sys_ts'
        IMMOBILE_DIR = r"D:\pre-release\data\immobile"
        PORTABLE_DIR = r"D:\pre-release\data\portable"
        FRAME_TIMESTAMP_DELTA_THRESHOLD_MS = 33

        TARGETS = list(filter(lambda x: osp.isdir(x), [osp.join(IMMOBILE_DIR, x) for x in os.listdir(IMMOBILE_DIR)] + [osp.join(PORTABLE_DIR, x) for x in os.listdir(PORTABLE_DIR)]))
        print(TARGETS)
        with tqdm.tqdm(total=len(TARGETS)) as pbar:
            for target in TARGETS:
                process_recording(target, FRAME_TIMESTAMP_DELTA_THRESHOLD_MS)
                pbar.update(1)


if __name__ == '__main__':
    main()
    # process_recording(r"D:\pre-release\data\immobile\bottle-014-1")
