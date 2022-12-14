import cv2
import numpy as np

from norfair import Detection, Tracker, Video, draw_tracked_objects


# Set up Detectron2 object detector
# detector = DefaultPredictor(cfg)

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


# Norfair
video = Video(input_path="video.mp4")
tracker = Tracker(distance_function=euclidean_distance, distance_threshold=20)

for frame in video:
    detections = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detections = [Detection(p) for p in detections['instances'].pred_boxes.get_centers().cpu().numpy()]
    tracked_objects = tracker.update(detections=detections)
    draw_tracked_objects(frame, tracked_objects)
    video.write(frame)
