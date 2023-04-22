import os.path as osp
import sys

sys.path.append('.')
from markit_processing.datamodels import IMUStreamModel, RealsenseStreamModel
import streamlit as st
import yaml
import cv2
import matplotlib.pyplot as plt


if 'next_recording_index' not in st.session_state.keys():
    st.session_state['next_recording_index'] = 0
dataset_base_dir = st.text_input('dataset_base_dir', r'D:\pre-release')
dataset_type = st.selectbox('dataset_type', ['immobile', 'portable'])
with open(osp.join(dataset_base_dir, 'metadata', dataset_type, 'index_patched.yaml'), 'r') as f:
    dataset_metadata_raw = yaml.load(f, Loader=yaml.SafeLoader)

imu_friendly_name_reverse_mapping = {v: k for k, v in dataset_metadata_raw['articulated']['imu_friendly_name_mapping'].items()}

recording_index = st.selectbox('recording_index', list(range(len(dataset_metadata_raw['articulated']['targets']))), st.session_state['next_recording_index'])
recording_name = list(map(lambda x: x['recordings'][0], dataset_metadata_raw['articulated']['targets']))[recording_index]
st.write("recording_name:", recording_name)
recording_path = osp.join(dataset_base_dir, 'data', dataset_type, recording_name)

imu_stream_model = IMUStreamModel(osp.join(recording_path, 'imu'))
imu_stream_model.load()
imu_toggled = list(map(lambda x: imu_friendly_name_reverse_mapping[x], imu_stream_model.path_to_recordings.keys()))


fig = plt.figure(figsize=(18, 24))

imu_device_ids = list(imu_stream_model.recordings.keys())
for i in range(len(imu_device_ids)):
    ax = fig.add_subplot(len(imu_device_ids), 1, i + 1)
    ax.plot(imu_stream_model.recordings[imu_device_ids[i]]['timestamp'],imu_stream_model.recordings[imu_device_ids[i]]['gyro_x'])
    ax.set_title(imu_friendly_name_reverse_mapping[imu_device_ids[i]])
    ax.set_ylabel('gyro_x')

st.pyplot(fig)

if len(dataset_metadata_raw['articulated']['targets'][recording_index]['imu_dependency']) == 0:
    imu_dependency = st.multiselect("imu_dependency", imu_toggled, default=imu_toggled)
else:
    st.write("imu_dependency:", dataset_metadata_raw['articulated']['targets'][recording_index]['imu_dependency'])
    imu_dependency = st.multiselect("imu_dependency", imu_toggled, default=dataset_metadata_raw['articulated']['targets'][recording_index]['imu_dependency'])

print(imu_dependency)
print(imu_toggled)
print(dataset_metadata_raw['articulated']['targets'][recording_index]['imu_dependency'])
realsense_stream_model = RealsenseStreamModel(osp.join(recording_path, 'realsense'))
realsense_stream_model.load()
realsense_visualization_key = st.selectbox("realsense_visualization_key", realsense_stream_model.camera_friendly_names)
num_of_frames = len(realsense_stream_model.selected_frames['filenames'][realsense_visualization_key]['color'])
selected_frame_idx = st.slider("selected_frame_idx", 0, num_of_frames - 1, 0)
selected_frame_path = osp.join(recording_path, 'realsense', realsense_visualization_key, 'color',
                               realsense_stream_model.selected_frames['filenames'][realsense_visualization_key]['color'][selected_frame_idx])
st.header("realsense_preview")
st.image(cv2.cvtColor(cv2.imread(selected_frame_path), cv2.COLOR_BGR2RGB))
if st.button('Save'):
    dataset_metadata_raw['articulated']['targets'][recording_index]['imu_toggled'] = imu_toggled
    dataset_metadata_raw['articulated']['targets'][recording_index]['imu_dependency'] = imu_dependency
    print(dataset_metadata_raw['articulated']['targets'][recording_index]['imu_toggled'], dataset_metadata_raw['articulated_kit']['targets'][recording_index]['imu_dependency'])
    with open(osp.join(dataset_base_dir, 'metadata', dataset_type, 'index_patched.yaml'), 'w') as f:
        yaml.dump(dataset_metadata_raw, f, sort_keys=False)
    st.write("Saved to index_patched.yaml")
    st.session_state['next_recording_index'] = min(len(dataset_metadata_raw['articulated']['targets']) - 1,recording_index + 1)
    st._rerun()
else:
    pass
print('done')
