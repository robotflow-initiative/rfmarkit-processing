import sys

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('.')
from internal.datamodels import PreReleaseIMUStreamModel, PreReleaseRealsenseStreamModel
import os.path as osp
import os
import cv2
from internal.utils.algorithm import IMUAlgorithm
import datetime
import json

st.set_option('deprecation.showPyplotGlobalUse', False)

METHOD = 'ts'
FORMAT = 'jpeg'

recording_base_dir_ = st.text_input('recording_base_dir', r'D:\pre-release\data')  # FIXME: change to your local path
recording_type = st.selectbox('recording_type', [osp.basename(x) for x in os.listdir(recording_base_dir_)])
recording_base_dir = osp.join(recording_base_dir_, recording_type)
recording_name = ""
try:
    recording_name = st.selectbox('recording_path', [osp.basename(x) for x in os.listdir(recording_base_dir)])
except FileNotFoundError:
    st.write("No recording found, check recording_base_dir")
    st.stop()
    exit(1)

imu_path = osp.join(recording_base_dir, recording_name, "imu")
realsense_path = osp.join(recording_base_dir, recording_name, "realsense")
imu_dataset = PreReleaseIMUStreamModel(imu_path)
realsense_dataset = PreReleaseRealsenseStreamModel(realsense_path)

alignment = None
if os.path.exists(osp.join(recording_base_dir, recording_name, "imu_realsense_alignment.json")):
    try:
        with open(osp.join(recording_base_dir, recording_name, "imu_realsense_alignment.json"), 'r') as f:
            alignment = json.load(f)
            st.write("Alignment loaded from file")
            st.write(alignment)
    except:
        st.write("Failed to load alignment from file")

try:
    imu_dataset.load()
    realsense_dataset.load()
    # st.write("Loaded dataset")
except:
    st.write("Failed to load dataset")
    st.stop()
    exit(0)

device_ids = list(imu_dataset.recordings.keys())
# print(device_ids)
# st.write("Device IDs", device_ids)

# st.write("Files", dataset.recordings[device_ids].files)

imu_timestamp_min = max([x['timestamp'].min() for x in imu_dataset.recordings.values()])
imu_timestamp_max = min([x['timestamp'].max() for x in imu_dataset.recordings.values()])
imu_timestamp_min = datetime.datetime.fromtimestamp(imu_timestamp_min / 1e6)
imu_timestamp_max = datetime.datetime.fromtimestamp(imu_timestamp_max / 1e6)

imu_start_time = st.slider("imu_start_time",
                           imu_timestamp_min,
                           imu_timestamp_max,
                           imu_timestamp_min if alignment is None else datetime.datetime.fromtimestamp(alignment['imu_start_timestamp_us'] / 1e6),
                           datetime.timedelta(seconds=0.01),
                           format="hh:mm:ss:SS")
imu_stop_time = st.slider("imu_stop_time",
                          imu_timestamp_min,
                          imu_timestamp_max,
                          imu_timestamp_max if alignment is None else datetime.datetime.fromtimestamp(alignment['imu_stop_timestamp_us'] / 1e6),
                          datetime.timedelta(seconds=0.01),
                          format="hh:mm:ss:SS")

imu_start_timestamp = int(imu_start_time.timestamp() * 1e6)
imu_stop_timestamp = int(imu_stop_time.timestamp() * 1e6)

# st.write("imu_start_timestamp", imu_start_timestamp)
# st.write("imu_stop_timestamp", imu_stop_timestamp)

imu_visualisation_key = st.selectbox("imu_visualisation_key", ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z'])
clipped_sequences_for_visualisation = {
    "data": {
        k: v[imu_visualisation_key][(v['timestamp'] >= imu_start_timestamp) & (v['timestamp'] <= imu_stop_timestamp)] for k, v in imu_dataset.recordings.items()
    },
    "timestamp": {
        k: v['timestamp'][(v['timestamp'] >= imu_start_timestamp) & (v['timestamp'] <= imu_stop_timestamp)] for k, v in imu_dataset.recordings.items()
    }
}



# fig = IMUAlgorithm.visualize_nd([clipped_sequences_for_visualisation['data'][device_id] for device_id in imu_dataset.recordings.keys()],
#                                 [clipped_sequences_for_visualisation['timestamp'][device_id] for device_id in imu_dataset.recordings.keys()],
#                                 title="visualisation_key")
# ax = fig.get_subplot(111)
fig = plt.figure(figsize=(18, 10))

for i in range(len(clipped_sequences_for_visualisation['data'].keys())):
    ax = fig.add_subplot(len(clipped_sequences_for_visualisation['data'].keys()), 1, i + 1)
    ax.plot(clipped_sequences_for_visualisation['timestamp'][device_ids[i]], clipped_sequences_for_visualisation['data'][device_ids[i]])
    ax.set_title(device_ids[i])
    ax.set_ylabel(imu_visualisation_key)
# ax = fig.add_subplot(111)
# for i in range(n):
#     ax.scatter(timestamp[i], data[i], s=2)
# ax.set_title(title)



st.pyplot(fig)
realsense_visualization_key = st.selectbox("realsense_visualization_key", list(realsense_dataset.recordings.keys()))
realsense_timestamps = np.array([int(x.split('_')[1]) for x in realsense_dataset.selected_frames['filenames'][realsense_visualization_key]['color']]) * 1e3 # convert to us

realsense_minus_imu = st.number_input("realsense_minus_imu (s)", -1000., 1000., 0. if alignment is None else alignment['realsense_minus_imu_us'] / 1e6, 0.1) * 1e6

realsense_start_idx = 0
realsense_stop_idx = len(realsense_timestamps) - 1

try:
    realsense_start_idx = list(np.where(realsense_timestamps >= imu_start_timestamp + realsense_minus_imu))[0][0]
    realsense_stop_idx = list(np.where(realsense_timestamps <= imu_stop_timestamp + realsense_minus_imu))[0][-1]
except:
    st.write("No related realsense frame")
    st.stop()
    exit(1)

print(realsense_start_idx, realsense_stop_idx)
realsense_start_frame_filename = realsense_dataset.selected_frames['filenames'][realsense_visualization_key]['color'][realsense_start_idx]
realsense_stop_frame_filename = realsense_dataset.selected_frames['filenames'][realsense_visualization_key]['color'][realsense_stop_idx]
realsense_start_frame_path = osp.join(realsense_path, realsense_visualization_key, "color", realsense_start_frame_filename)
realsense_stop_frame_path = osp.join(realsense_path, realsense_visualization_key, "color", realsense_stop_frame_filename)

print(realsense_start_frame_path, realsense_stop_frame_path)

col1, col2 = st.columns(2)
with col1:
   st.header("realsense_start")
   st.image(cv2.cvtColor(cv2.imread(realsense_start_frame_path), cv2.COLOR_BGR2RGB))
   if realsense_start_idx <= 0:
       st.write("realsense_stop_idx is out of range")
   else:
       st.write("realsense_start_idx", realsense_start_idx)


with col2:
   st.header("realsense_end")
   st.image(cv2.cvtColor(cv2.imread(realsense_stop_frame_path), cv2.COLOR_BGR2RGB))
   if realsense_stop_idx >= len(realsense_timestamps):
       st.write("realsense_stop_idx is out of range")
   else:
       st.write("realsense_stop_idx", realsense_stop_idx)

if st.button('Save'):
    with open( osp.join(recording_base_dir, recording_name,"imu_realsense_alignment.json"), "w") as f:
        json.dump({
            "imu_start_timestamp_us": imu_start_timestamp,
            "imu_stop_timestamp_us": imu_stop_timestamp,
            "realsense_start_idx": int(realsense_start_idx),
            "realsense_stop_idx": int(realsense_stop_idx),
            "realsense_minus_imu_us": realsense_minus_imu
        }, f, indent=4)
    st.write('Parameters saved')
else:
    pass