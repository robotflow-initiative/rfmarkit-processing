import copy
import io
import os.path as osp

import cv2
import cv2.aruco as aruco
import numpy as np
import yaml
from realsense_recorder.common import new_realsense_camera_system_from_config, RealsenseSystemModel

# INIT_DIST = np.array(([[0, 0, 0, 0, 0.1]]))
INIT_DIST = np.array([
            0.1927826544288516,
            -0.34972530095573834,
            0.011612480526787846,
            -0.00393533140166019,
            -2.9216752723525734
        ])
MARKER_LENGTH = 0.02
MARKER_POINTS = np.array([[-MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0],
                          [MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0],
                          [MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0],
                          [-MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0]])

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


# num = 0
def process_one_frame(frame, mtx, dist, font=cv2.FONT_HERSHEY_SIMPLEX):
    h1, w1 = frame.shape[:2]

    # 读取摄像头画面
    # 纠正畸变
    outputFrame = copy.deepcopy(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    '''
    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
    '''
    # 使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)


    # print(ids)

    # id found
    if ids is not None:
        aruco.drawDetectedMarkers(outputFrame, corners, ids)
        for corner in corners:
            _, rvec, tvec = cv2.solvePnP(MARKER_POINTS, corner[0], mtx, dist)

            # print(dist)

            cv2.drawFrameAxes(outputFrame, mtx, dist, rvec, tvec, 0.03)
            print(rvec, "\n")
            ###### DRAW ID #####
            cv2.putText(outputFrame, "Id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


    return outputFrame

#
# cfg_str = """
# realsense:
#   cameras:
#   - color:
#     - exposure: -1
#       format: rs.format.bgra8
#       fps: 30
#       height: 1080
#       width: 1920
#     depth: []
#     endpoint: {}
#     imu: []
#     product_id: 0B64
#     product_line: L500
#     ref: 1
#     sn: f0220485
#     use_depth: false
#   system:
#     base_dir: ./realsense_data
#     frame_queue_size: 100
#     interactive: false
#     interval_ms: 0
#     use_bag: false
# """


cfg_str = """
realsense:
  cameras:
  - color:
    - exposure: -1
      format: rs.format.bgra8
      fps: 30
      height: 720
      width: 1280
    depth: []
    endpoint: {}
    product_id: 0B07
    product_line: D400
    ref: 0
    sn: 049322071061
  system:
    base_dir: ./realsense_data
    frame_queue_size: 100
    interactive: false
    interval_ms: 0
    use_bag: false
"""
def main(output_path):
    cfg = yaml.load(io.StringIO(cfg_str), yaml.SafeLoader)
    sys = new_realsense_camera_system_from_config(RealsenseSystemModel, cfg['realsense'], None)
    print(sys.cameras)
    cam = sys.cameras[0]
    cam.open()
    cam.start()
    mtx = np.array(cam.intrinsics_matrix)
    idx = 0
    while True:
        color_image, depth_image, ts, sys_ts, frame_counter = cam.get_frames()
        # processed_frame = process_one_frame(color_image[:,:,:3], mtx, INIT_DIST)
        processed_frame = color_image
        # cv2.imshow("frame", processed_frame)

        key = cv2.waitKey(1)

        cv2.imwrite(osp.join(output_path, "frame_{:05d}.jpg".format(idx)), processed_frame)

        if key == 27:  # 按esc键退出
            print('esc break...')
            cv2.destroyAllWindows()
            break
        idx += 1


main(r'C:\Users\liyutong\Downloads\video\swing')
