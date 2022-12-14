import cv2
import numpy as np
import tqdm

from internal.detector import OpenCVDetector
from utils import RealsenseCameraSystem, RealsenseCamera, get_directory_reader

# Hyper parameters
CONFIG_DETECTOR_BLUR_SIZE = 9  # Filter with cv2.GaussianBlur for better thresh segmentation
CONFIG_DETECTOR_THRESH = 24  # Brightness thresh
CONFIG_DETECTOR_N_ERODE_ITERATIONS: int = 1  # Erode intensity
CONFIG_DETECTOR_ERODE_SIZE: int = 4  # Structure size, 2 or 3 is OK
CONFIG_DETECTOR_DEBUG = True
CONFIG_FG_DETECTION_ENABLED = False
CONFIG_FG_HISTORY = 5
CONFIG_FG_VAR_THRESHOLD = 16
CONFIG_READER_N_PRELOAD = 4

CONFIG_TRACKER_MINIMAL_DISTANCE = 32

# CONFIG_CAMERA_MASK = ["011422071122", "045322072962"]
CONFIG_CAMERA_MASK = ["011422071122"]
# CONFIG_CAMERA_MASK = ["045322072962"]


def _detection_job(d: OpenCVDetector, cam: RealsenseCamera):
    reader = get_directory_reader(cam.path_to_color_stream, "color_bmp", CONFIG_READER_N_PRELOAD)
    with tqdm.tqdm(range(len(reader))) as pbar:
        prev_frame = None
        while not reader.eof:
            frame, meta = reader.next()
            if frame is None:
                cv2.destroyAllWindows()
                break

            fusion_frame = np.maximum(prev_frame, frame) if prev_frame is not None else frame  # Run detection on mixture of two frames
            rendered_frame, res, _ = d.process_frame(fusion_frame, CONFIG_DETECTOR_DEBUG)
            pbar.update()

            if CONFIG_DETECTOR_DEBUG:
                cv2.imshow('Analyse', rendered_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            prev_frame = frame


if __name__ == '__main__':
    RECORDING_DIR = r"/Users/liyutong/Downloads/bulu-0-0/"

    # INPUT: MEASUREMENT_DIR
    # OUTPUT: analyse_results

    CamSys = RealsenseCameraSystem(RECORDING_DIR, mask=CONFIG_CAMERA_MASK)
    Det = OpenCVDetector(thresh=CONFIG_DETECTOR_THRESH,
                         erode_size=CONFIG_DETECTOR_ERODE_SIZE,
                         n_erode_iterations=CONFIG_DETECTOR_N_ERODE_ITERATIONS,
                         blur_size=CONFIG_DETECTOR_BLUR_SIZE,
                         fg_detection_enabled=CONFIG_FG_DETECTION_ENABLED,
                         fg_history=CONFIG_FG_HISTORY,
                         fg_var_threshold=CONFIG_FG_VAR_THRESHOLD)
    _detection_job(Det, CamSys.cameras[0])

    #
    # measurement_ctx = TrackerUtils.list_directory(MEASUREMENT_DIR)
    # resources: Dict = TrackerUtils.open_resources(measurement_ctx)
    # analyse_results: Dict[str, List[Dict]] = {}
    # track_data: Dict[str, List[List]] = {}
    # # [
    # #  {
    # #  "timestamp": float
    # #  "centers": [
    # #                {"x": float, # x coordinate
    # #                 "y": float, # y coordinate
    # #                 "r": float, # radius,
    # #                 "t": float, # time,
    # #                 "frame_no": int  # frame no
    # #                },...
    # #              ]
    # #  },...
    # # ]
    # #
    # for camera_sn in resources.keys():
    #
    #     camera_resource = resources[camera_sn]
    #     analyse_results[camera_sn] = []
    #     cap = camera_resource['cap']  # Read video
    #     frames = []
    #     frame_num = 0
    #
    #     with tqdm.tqdm(range(int(camera_resource['frame_count']))) as pbar:
    #         while cap.isOpened() and frame_num < len(camera_resource['timestamp']):
    #             ret, origin_frame = cap.read()
    #             if not ret:
    #                 cv2.destroyAllWindows()
    #                 break
    #             frames.append(origin_frame)
    #             center_frame, res = Tracker.process_frame(origin_frame, THRESH, ERODE_SIZE, ERODE_ITERATIONS, BLUR_SIZE,
    #                                                       float(camera_resource['timestamp'][frame_num]), frame_num)
    #             frame_num += 1
    #             analyse_results[camera_sn].append(res)
    #             pbar.update()
    #
    #             if VISUALIZE:
    #                 cv2.imshow('Analyse', center_frame)
    #                 if cv2.waitKey(1) & 0xFF == ord('q'):
    #                     cv2.destroyAllWindows()
    #                     break
    #
    #     # print(analyse_results)
    #     # INPUT: analyse_results
    #     # OUTPUT: List[TrackSequence]
    #     finished_track_sequences = Tracker.track(analyse_results[camera_sn], TRACK_MINIMAL_DISTANCE)
    #
    #     # INPUT: List of TrackSequences
    #     # OUTPUT: Index of imu
    #     [seq.process_track(frames) for seq in finished_track_sequences]
    #
    #     track_data[camera_sn] = [[] for _ in range(frame_num)]
    #     for seq in finished_track_sequences:
    #         for center in seq.seq:
    #             x, y, r, t, no = center['x'], center['y'], int(center['r']), center['t'], center['frame_no']
    #             track_data[camera_sn][no].append({'x': x, 'y': y, 'r': r, 'idx': seq.idx, 'no': no, 'timestamp': t})
    #
    #     if VISUALIZE:
    #         if not os.path.exists('example/out'):
    #             os.makedirs('example/out')
    #         for frame_no in range(frame_num):
    #             frame = frames[frame_no]
    #             for point in track_data[camera_sn][frame_no]:
    #                 # 在图像上绘制轮廓及中心
    #                 cv2.circle(frame, (point['x'], point['y']), int(point['r']), (255, 255, 255), 1)
    #                 cv2.putText(frame, str(point['idx']), (point['x'] - 20, point['y'] - 20),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    #             cv2.imshow('demo_control', frame)
    #             cv2.imwrite(f'./out/{frame_num}.jpg', frame)
    #             frame_num += 1
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 cv2.destroyAllWindows()
    #                 break
    #     cv2.destroyAllWindows()
    #
    # # Collect date for visualisation
    # track_data_json_output = os.path.join(MEASUREMENT_DIR, f'track_data.json')
    # with open(track_data_json_output, 'w') as f:
    #     json.dump(track_data, f)

    print('finish')
