import sys

import numpy as np
import streamlit as st

sys.path.append('.')
from internal.datamodels import IMUStreamModel, RealsenseStreamModel
import os.path as osp
import os
import cv2
import json
import yaml

st.set_option('deprecation.showPyplotGlobalUse', False)

METHOD = 'ts'
FORMAT = 'jpeg'
CAMERA_REFERENCE_MAP = {
    "r22": "r08",
    "r69": "r85"
}
if 'next_objects_to_reid' not in st.session_state.keys():
    st.session_state['next_objects_to_reid'] = 0

recording_base_dir_ = st.text_input('根目录（无需修改）', r'D:\pre-release\data')
recording_type = st.selectbox('实验类型', [osp.basename(x) for x in os.listdir(recording_base_dir_)])
recording_base_dir = osp.join(recording_base_dir_, recording_type)
recording_name = ""
try:
    # recording_name = "bottle-014-1"
    recording_name = st.selectbox('实验名称', [osp.basename(x) for x in os.listdir(recording_base_dir)])
except FileNotFoundError:
    st.write("No recording found, check recording_base_dir")
    st.stop()
    exit(1)

imu_path = osp.join(recording_base_dir, recording_name, "imu")
realsense_path = osp.join(recording_base_dir, recording_name, "realsense")

reid_path = osp.join(realsense_path, "reid.json")
if osp.exists(reid_path):
    try:
        current_reid_mapping = json.load(open(reid_path, "r"))
    except json.decoder.JSONDecodeError:
        current_reid_mapping = None
else:
    current_reid_mapping = None

imu_dataset = IMUStreamModel(imu_path)
realsense_dataset = RealsenseStreamModel(realsense_path)
realsense_dataset.load()
imu_dataset.load()
dataset_metadata_raw = yaml.load(open(osp.join(recording_base_dir_, "../metadata", recording_type, "index.yaml"), "r"), Loader=yaml.SafeLoader)['articulated_kit']
candidate_imu_device_ids = list(imu_dataset.recordings.keys())
candidate_imu_device_friendly_names = ["-"]
candidate_imu_device_friendly_names += list(filter(lambda x: dataset_metadata_raw['imu_friendly_name_mapping'][x] in candidate_imu_device_ids, dataset_metadata_raw['imu_friendly_name_mapping']))
candidate_imu_device_friendly_names += ["?"]  # for unknown

if current_reid_mapping is not None:
    _rest_objects = 0
    for v in realsense_dataset.led_tracking_result['result'].keys():
        if v in current_reid_mapping['reid'].keys():
            _rest_objects += max(0, len(realsense_dataset.led_tracking_result['result'][v].keys()) - len(list(filter(lambda x: x != "-", current_reid_mapping['reid'][v].values()))))
        else:
            _rest_objects += len(realsense_dataset.led_tracking_result['result'][v].keys())
    if _rest_objects <= 0:
        st.write("所有物体都已经标注完毕")
    else:
        st.write("还有{}个物体需要标注".format(_rest_objects))

# view_to_process = list(realsense_dataset.led_tracking_result['meta']['cameras'])[0]
view_to_process = st.selectbox('视角', list(realsense_dataset.led_tracking_result['meta']['cameras']) if recording_type == "immobile" else ["r22"])

# objects_to_reid = list(realsense_dataset.led_tracking_result['result'][view_to_process].keys())[0]
objects_to_reid = st.selectbox('待标注对象ID', sorted(list(realsense_dataset.led_tracking_result['result'][view_to_process].keys()), key=lambda x: int(x)),
                               index=max(0, min(len(list(realsense_dataset.led_tracking_result['result'][view_to_process].keys())) - 1, st.session_state['next_objects_to_reid'])))
# st.session_state['next_objects_to_reid'] = min(len(list(realsense_dataset.led_tracking_result['result'][view_to_process].keys())) - 1, int(objects_to_reid) + 1)

object_frame_meta = realsense_dataset.led_tracking_result['result'][view_to_process][objects_to_reid][0]
center = (int(object_frame_meta['position'][0][0]), int(object_frame_meta['position'][0][1]))
object_frame_index_relative_to_sequence = [int(x['frame_idx']) for x in realsense_dataset.recordings[view_to_process].color.metadata].index(
    int(object_frame_meta['meta']['frame_idx']))  # - int(realsense_dataset.recordings[view_to_process].color.metadata[0]['frame_idx'])
object_frame = cv2.imread(osp.join(realsense_dataset.path_to_stream, view_to_process, "color", object_frame_meta['meta']['basename']))
object_reference_frame = cv2.imread(osp.join(realsense_dataset.path_to_stream, CAMERA_REFERENCE_MAP[view_to_process], "color",
                                             realsense_dataset.recordings[CAMERA_REFERENCE_MAP[view_to_process]].color.metadata[object_frame_index_relative_to_sequence]['basename']))
cv2.circle(object_frame, (int(center[0]), int(center[1])), 20, (255, 255, 255), 3)
cv2.putText(object_frame, "?", (int(center[0]) - 40, int(center[1]) - 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
# col1, col2 = st.columns(2)
# with col1:
st.header(view_to_process)
# st.image(cv2.cvtColor(object_frame, cv2.COLOR_BGR2RGB))
# with col2:
# st.header( CAMERA_REFERENCE_MAP[view_to_process])
# st.image(cv2.cvtColor(object_reference_frame, cv2.COLOR_BGR2RGB))
cols, rows, _ = object_frame.shape
minimum_brightness = st.slider('最小亮度', 0., 1., 0.66)
brightness = np.sum(object_frame) / (255 * cols * rows)
ratio = min(1, brightness / minimum_brightness)
st.image(cv2.cvtColor(cv2.convertScaleAbs(object_frame, alpha=1 / ratio, beta=0), cv2.COLOR_BGR2RGB))

new_index = 0
if current_reid_mapping is not None:
    if view_to_process in current_reid_mapping['reid'].keys():
        if objects_to_reid in current_reid_mapping['reid'][view_to_process].keys():
            new_index = max(0, candidate_imu_device_friendly_names.index(current_reid_mapping['reid'][view_to_process][objects_to_reid]))

new_id = st.radio('新的ID', candidate_imu_device_friendly_names, index=new_index, horizontal=1)
if st.button('保存'):
    if current_reid_mapping is None:
        current_reid_mapping = {
            "reid": {
                view_to_process: {
                    objects_to_reid: new_id
                }
            }
        }
    else:
        print(current_reid_mapping)
        if view_to_process not in current_reid_mapping['reid'].keys():
            current_reid_mapping['reid'][view_to_process] = {}
        current_reid_mapping['reid'][view_to_process][objects_to_reid] = new_id

    with open(reid_path, "w") as f:
        json.dump(current_reid_mapping, f, indent=4, sort_keys=True)
    st.write('Parameters saved')
    st.session_state['next_objects_to_reid'] = min(len(list(realsense_dataset.led_tracking_result['result'][view_to_process].keys())) - 1,
                                                   max(0, sorted(list(realsense_dataset.led_tracking_result['result'][view_to_process].keys()), key=lambda x: int(x)).index(objects_to_reid) + 1))
    st._rerun()
else:
    if st.button('上一个'):
        st.session_state['next_objects_to_reid'] = min(len(list(realsense_dataset.led_tracking_result['result'][view_to_process].keys())) - 1,
                                                       max(0, sorted(list(realsense_dataset.led_tracking_result['result'][view_to_process].keys()), key=lambda x: int(x)).index(objects_to_reid) - 1))
        print(st.session_state['next_objects_to_reid'], sorted(list(realsense_dataset.led_tracking_result['result'][view_to_process].keys()), key=lambda x: int(x)).index(objects_to_reid))
        st._rerun()

    if st.button('下一个'):
        st.session_state['next_objects_to_reid'] = min(len(list(realsense_dataset.led_tracking_result['result'][view_to_process].keys())) - 1,
                                                       max(0, sorted(list(realsense_dataset.led_tracking_result['result'][view_to_process].keys()), key=lambda x: int(x)).index(objects_to_reid) + 1))
        st._rerun()

if current_reid_mapping is not None:
    if view_to_process in current_reid_mapping['reid'].keys():
        if objects_to_reid in current_reid_mapping['reid'][view_to_process].keys():
            st.write("当前标注结果: {}".format(current_reid_mapping['reid'][view_to_process][objects_to_reid]))
            # print(candidate_imu_device_friendly_names.index(current_reid_mapping['reid'][view_to_process][objects_to_reid]))
            new_index = max(0, candidate_imu_device_friendly_names.index(current_reid_mapping['reid'][view_to_process][objects_to_reid]))
        else:
            st.write("无结果")
