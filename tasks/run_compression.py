import datetime
import glob
import os
import os.path as osp
import shutil
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import cv2
import numpy as np
import tqdm
from realsense_recorder.io import get_directory_reader
from rich.console import Console

console = None


def _read_color(path: str) -> Any:
    """
    Read anything, from a path
    :param path:
    :return: numpy.ndarray, RGB frame
    """
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _compress_color_folder(input_folder, output_folder, n_prefetch):
    reader = get_directory_reader(input_folder, 'color_bmp', num_preload=n_prefetch, read_function=_read_color)
    with tqdm.tqdm(range(len(reader))) as pbar:
        while not reader.eof:
            frame, meta = reader.next()
            frame_basename_without_ext = osp.splitext(osp.basename(meta['basename']))[0]
            cv2.imwrite(osp.join(output_folder, frame_basename_without_ext + ".jpeg"), frame)
            pbar.update()


def _compress_depth_folder(input_folder, output_folder, n_prefetch):
    reader = get_directory_reader(input_folder, 'depth_npy', num_preload=n_prefetch)
    with tqdm.tqdm(range(len(reader))) as pbar:
        while not reader.eof:
            frame, meta = reader.next()
            frame_basename_without_ext = osp.splitext(osp.basename(meta['basename']))[0]
            np.savez_compressed(osp.join(output_folder, frame_basename_without_ext + ".npz"), frame)
            pbar.update()


def compress_record(input_recording: str, output_base_dir: str, n_prefetch=16):
    input_recording_name = osp.basename(input_recording)

    console.log(f"Input recording: {input_recording}")

    # Prepare output directory
    OUTPUT_RECORDING = osp.join(output_base_dir, input_recording_name)
    if osp.exists(OUTPUT_RECORDING):
        console.log(f"Output recording: {OUTPUT_RECORDING} already exists. Delete it first.")

    os.mkdir(OUTPUT_RECORDING) if not osp.exists(OUTPUT_RECORDING) else None
    os.mkdir(osp.join(OUTPUT_RECORDING, "realsense")) if not osp.exists(osp.join(OUTPUT_RECORDING, "realsense")) else None
    camera_folders = list(
        map(lambda x: os.path.basename(x),
            list(filter(lambda x: os.path.isdir(x),
                        glob.glob(osp.join(input_recording, "realsense", "*"))
                        )
                 )
            )
    )
    # console.log(f"Camera folders: {camera_folders}")
    [os.mkdir(osp.join(OUTPUT_RECORDING, "realsense", camera_folder)) for camera_folder in camera_folders if not osp.exists(osp.join(OUTPUT_RECORDING, "realsense", camera_folder))]
    [os.mkdir(osp.join(OUTPUT_RECORDING, "realsense", camera_folder, 'color')) for camera_folder in camera_folders if not osp.exists(osp.join(OUTPUT_RECORDING, "realsense", camera_folder, 'color'))]
    [os.mkdir(osp.join(OUTPUT_RECORDING, "realsense", camera_folder, 'depth')) for camera_folder in camera_folders if not osp.exists(osp.join(OUTPUT_RECORDING, "realsense", camera_folder, 'depth'))]

    system_meta_data_files = list(filter(lambda x: os.path.isfile(x), glob.glob(osp.join(input_recording, "realsense", "*"))))
    # console.log(f"Meta data: {system_meta_data_files}")

    try:
        shutil.rmtree(osp.join(OUTPUT_RECORDING, "imu"))
    except:
        pass
    shutil.copytree(osp.join(input_recording, "imu"), osp.join(OUTPUT_RECORDING, "imu"))
    [shutil.copyfile(meta_data_file, osp.join(OUTPUT_RECORDING, "realsense", osp.basename(meta_data_file))) for meta_data_file in system_meta_data_files]
    [shutil.copyfile(osp.join(input_recording, "realsense", camera_folder, "realsense_intrinsic.json"), osp.join(OUTPUT_RECORDING, "realsense", camera_folder, "realsense_intrinsic.json")) for
     camera_folder in camera_folders]

    folder_compression_mapping_color = {
        osp.join(input_recording, "realsense", camera_folder, 'color'): osp.join(OUTPUT_RECORDING, "realsense", camera_folder, 'color') for camera_folder in camera_folders
    }
    folder_compression_mapping_depth = {
        osp.join(input_recording, "realsense", camera_folder, 'depth'): osp.join(OUTPUT_RECORDING, "realsense", camera_folder, 'depth') for camera_folder in camera_folders
    }

    console.log("Folder compression mapping:")
    console.log(folder_compression_mapping_color)
    console.log(folder_compression_mapping_depth)

    with ProcessPoolExecutor(max_workers=4) as pool:
        ctx = []
        for _input_folder, _output_folder in folder_compression_mapping_color.items():
            console.log(f"Compressing color {_input_folder} to {_output_folder}")
            ctx.append(pool.submit(_compress_color_folder, _input_folder, _output_folder, n_prefetch))
        [_.result() for _ in ctx]

    # for _input_folder, _output_folder in folder_compression_mapping_color.items():
    #     console.log(f"Compressing color {_input_folder} to {_output_folder}")
    #     _compress_color_folder(_input_folder, _output_folder, n_prefetch)

    with ProcessPoolExecutor(max_workers=4) as pool:
        ctx = []
        for _input_folder, _output_folder in folder_compression_mapping_depth.items():
            console.log(f"Compressing depth {_input_folder} to {_output_folder}")
            ctx.append(pool.submit(_compress_depth_folder, _input_folder, _output_folder, n_prefetch))
        [_.result() for _ in ctx]

    #
    # for _input_folder, _output_folder in folder_compression_mapping_depth.items():
    #     console.log(f"Compressing depth {_input_folder} to {_output_folder}")
    #     _compress_depth_folder(_input_folder, _output_folder, n_prefetch)
    #
    # for _input_folder, _output_folder in folder_compression_mapping_color.items():
    #     console.log(f"Compressing color {_input_folder} to {_output_folder}")
    #     _compress_color_folder(_input_folder, _output_folder, n_prefetch)


def main():
    global console
    with open('./' + "run_compression_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log", 'w') as f:
        console = Console(file=f)

        # FIXME: REPLACE WITH YOUR OWN PATHS
        N_PREFETCH = 8
        IMMOBILE_INPUT_RECORDING_PATTERN = r"\\100.99.96.101\articulated_recording\pre-release\data\immobile\*"
        IMMOBILE_INPUT_RECORDING_LIST = list(filter(lambda x: osp.isdir(x), glob.glob(IMMOBILE_INPUT_RECORDING_PATTERN)))
        IMMOBILE_OUTPUT_DIR = r"D:\pre-release\data\immobile"

        PORTABLE_INPUT_RECORDING_PATTERN = r"\\100.99.96.101\articulated_recording\pre-release\data\portable\*"
        PORTABLE_INPUT_RECORDING_LIST = list(filter(lambda x: osp.isdir(x), glob.glob(PORTABLE_INPUT_RECORDING_PATTERN)))
        PORTABLE_OUTPUT_DIR = r"D:\pre-release\data\portable"

        ARGUMENT_LIST = [(x, IMMOBILE_OUTPUT_DIR, N_PREFETCH) for x in IMMOBILE_INPUT_RECORDING_LIST] + [(x, PORTABLE_OUTPUT_DIR, N_PREFETCH) for x in PORTABLE_INPUT_RECORDING_LIST]

        console.print(ARGUMENT_LIST)

        # compress_record(*ARGUMENT_LIST[0])

        for arg in ARGUMENT_LIST:
            try:
                compress_record(*arg)
            except Exception as e:
                console.log(e)


if __name__ == '__main__':
    # debug
    pass
    # main()
    # console = Console()
    # compress_record(r"\\100.99.96.101\articulated_recording\archived\pre-release-2022-12-09\data\portable\box-024-1", r"C:\Users\liyutong\Downloads")
