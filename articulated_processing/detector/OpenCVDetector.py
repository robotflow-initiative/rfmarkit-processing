import math
from typing import Tuple

import cv2
import imutils
import numpy as np


class OpenCVDetector(object):
    def __init__(self,
                 thresh: int = 40,
                 erode_size: int = 1,
                 n_erode_iterations: int = 2,
                 n_dilate_iterations: int = 2,
                 blur_size: int = 11,
                 fg_detection_enabled: bool = False,
                 fg_history: int = 100,
                 fg_var_threshold: int = 16):
        self.thresh = thresh
        self.erode_size = erode_size
        self.n_erode_iterations = n_erode_iterations
        self.n_dilate_iterations = n_dilate_iterations
        self.blur_size = blur_size

        if fg_detection_enabled:
            self.fgbg = cv2.createBackgroundSubtractorMOG2(history=fg_history, varThreshold=fg_var_threshold)
        else:
            self.fgbg = None
        self.fgbg_cache = np.empty(0)

        pass

    def process_frame(self,
                      frame: np.ndarray,
                      debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        frame: BGR format frame
        debug: bool flag

        Returns
        -------

        """
        debug_frame: np.ndarray = frame.copy()

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(cv2.inRange(hsv_frame, np.array([0,43,46]), np.array([10,255,255])),cv2.inRange(hsv_frame, np.array([156,43,46]), np.array([180,255,255])))
        blue_mask = cv2.inRange(hsv_frame, np.array([100,43,46]), np.array([124,255,255]))
        all_mask = cv2.bitwise_or(red_mask, blue_mask)
        frame = cv2.bitwise_and(frame, frame, mask=all_mask)
        frame = cv2.convertScaleAbs(frame,alpha=2, beta=20)

        if self.fgbg is not None:
            fgmask: np.ndarray = self.fgbg.apply(frame)
            self.fgbg_cache = np.maximum(self.fgbg_cache, fgmask) if self.fgbg_cache.size > 0 else fgmask
            frame = cv2.bitwise_or(frame, frame, mask=fgmask)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur_frame = cv2.GaussianBlur(gray_frame, (self.blur_size, self.blur_size), 0)
        thresh, binary_frame = cv2.threshold(blur_frame, self.thresh, 255, cv2.THRESH_BINARY)
        binary_frame = cv2.erode(binary_frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.erode_size, self.erode_size)),
                                 iterations=self.n_erode_iterations)
        binary_frame = cv2.dilate(binary_frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.erode_size, self.erode_size)),
                                  iterations=self.n_dilate_iterations)

        contours, hierarchy = cv2.findContours(binary_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours((contours, hierarchy))

        # 遍历轮廓集

        centers = np.empty(shape=(len(cnts), 3))
        for idx, c in enumerate(cnts):
            M = cv2.moments(c)
            centers[idx, :] = int(M["m10"] / (1e-4 + M["m00"])), int(M["m01"] / (1e-4 + M["m00"])), math.sqrt(M['m00'])  # cX, cY, r
            # http://edu.pointborn.com/article/2021/11/19/1709.html

        if debug:
            # 在图像上绘制轮廓及中心
            for center in centers:
                cv2.circle(debug_frame, (int(center[0]), int(center[1])), int(center[2]), (255, 255, 255), 1)
                # cv2.putText(debug_frame, "center", (int(center[0]) - 20, int(center[1]) - 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return debug_frame, centers, self.fgbg_cache
