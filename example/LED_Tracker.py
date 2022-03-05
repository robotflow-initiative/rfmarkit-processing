import os
import cv2
import numpy as np
import argparse
import copy
from typing import List, Optional


def brightness_filter(img, thresh=100.0):
    brightness_img = np.sum(img, axis=-1) / 3
    thresh_mask = brightness_img > thresh
    brightness_img *= thresh_mask

    return brightness_img


def detect_LED(brightness_img, thresh=10):
    pixels = np.stack(np.nonzero(brightness_img), axis=1)
    pixels = np.stack([pixels[:, 1], pixels[:, 0]], axis=1)
    clusters = []
    centers = np.empty((0, 2))
    for pixel in pixels:
        dist_to_centers = ((pixel - centers) ** 2).sum(axis=1) ** 0.5
        dist_thresh_mask = dist_to_centers < thresh
        belong_clusters = np.nonzero(dist_thresh_mask)[0]
        if len(belong_clusters):
            idx = belong_clusters[0]
            N = len(clusters)
            centers[idx] = (centers[idx] * N + pixel) / (N + 1)
            clusters[idx].append(pixel)
        else:
            centers = np.concatenate([centers, pixel[None, :]])
            clusters.append([pixel])
    return centers, clusters


def expel_LED(centers, expelled, thresh=5):
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers, dtype=np.float)
    if not isinstance(expelled, np.ndarray):
        expelled = np.array(expelled, dtype=np.float)

    dist = centers[:, None, :] - expelled[None, :, :]
    dist = ((dist ** 2).sum(axis=2) ** 0.5).min(axis=1)
    mask = dist > thresh
    indices = np.nonzero(mask)[0]
    return np.take(centers, indices, axis=0)


class LED_Info:
    def __init__(self,
                 centers: List[Optional[np.ndarray]] = [],
                 timestamps: List[float] = [],
                 index: int = -1):
        self.centers = centers
        self.index = index

    def __len__(self):
        return len(self.centers)


class Center_Info:
    def __init__(self,
                 center: np.ndarray,
                 center_idx_in_following_frames: List[int] = [],
                 led_index: int = -1):
        self.center = center
        self.center_idx_in_following_frames = center_idx_in_following_frames
        self.led_index = led_index


class LED_Tracker:
    def __init__(self, images: List[np.ndarray],
                 timestamps: List[float],
                 expelled_pixels: List[List[float]],
                 brightness_thresh=70.,
                 detect_thresh=20.,
                 expel_thresh=5.,
                 motion_thresh=10.,
                 all_bits=10,
                 sync_bits=2,
                 cycle_time=0.1):
        """
        A LED Tracker for LED detecting and tracking.
        :param images: A List of NumPy arrays with shape (Height, Width, 3)
        :param timestamps: A List of float indicating the timestamp for each frame
        :param expelled_pixels: A list of pixels to be expelled. (Prevent these bright pixels from being recognized as LEDs)
        :param brightness_thresh: A float, indicates the threshold of brightness
        :param detect_thresh: A float, used to cluster the bright pixels
        :param expel_thresh: A float, used to expel some bright pixels which are not part of the desired LEDs
        :param motion_thresh: A float, used to judge whether two pixels in continuous frames belongs to the same LED.
        :param all_bits: A int, the number of all bits for encoding LED identity.
        :param sync_bits: A int, the number of sync bits for encoding LED identity.
        :param cycle_time: A float, the time of minimum LED blinking cycle.
        """
        # to make sure two different bright point won't correspond to the same LED in the last frame
        assert motion_thresh <= 0.5 * detect_thresh

        self.images = images
        self.timestamps = timestamps
        self.expelled_pixels = expelled_pixels
        self.brightness_thresh = brightness_thresh
        self.detect_thresh = detect_thresh
        self.expel_thresh = expel_thresh
        self.motion_thresh = motion_thresh
        self.all_bits = all_bits
        self.sync_bits = sync_bits
        self.cycle_time = cycle_time
        self._brightness_images = []
        self._centers_each_frame = []
        self._centers_info_list_each_frame = []
        self._LEDs = []
        self._neighbour_frames = []

        self.get_centers()
        self.get_neighboring_centers()
        self.get_LEDs()

    @staticmethod
    def get_diff_bit(bit_list: List[int]):
        """
        Finding the real bit in one manchester encoding cycle
        :param bit_list: List of int(0, 1)
        :return: int(0, 1)
        """
        num_bits = len(bit_list)
        for idx in range(num_bits):
            if sum(bit_list[:idx + 1]) == (idx + 1) and sum(bit_list[idx + 1:]) == 0:
                return 0
            elif sum(bit_list[:idx + 1]) == 0 and sum(bit_list[idx + 1:]) == (num_bits - 1 - idx):
                return 1
        return None # not valid

    def get_centers(self):
        for img_idx, img in enumerate(self.images):
            brightness_img = brightness_filter(img, self.brightness_thresh)
            self._brightness_images.append(brightness_img)
            centers, clusters = detect_LED(brightness_img)
            centers = expel_LED(centers, self.expelled_pixels)

            centers_img = np.zeros(brightness_img.shape + (3, ), dtype=np.uint8)
            center_infos = []
            for center in centers:
                center_infos.append(Center_Info(center))
                cv2.circle(centers_img, center.astype(int).tolist(), radius=10, color=(0, 0, 255), thickness=-1)

            cv2.namedWindow(f'frame {img_idx}', cv2.WINDOW_AUTOSIZE)
            centers_img = cv2.resize(centers_img.copy(), (centers_img.shape[1]//2, centers_img.shape[0]//2))
            cv2.imshow(f'frame {img_idx}', centers_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            self._centers_info_list_each_frame.append(center_infos)
            self._centers_each_frame.append(centers)

    def get_neighboring_centers(self):
        """
        Get all neighboring centers in the following frames for all the visible centers in each frame.
        :return: None
        """
        for i in range(len(self.images)):
            for j in range(i + 1, len(self.images)):
                self._neighbour_frames.append((i, j))
                centers_i = self._centers_each_frame[i]
                centers_j = self._centers_each_frame[j]
                centers_diff = ((centers_i[:, None, :] -
                                 centers_j[None, :, :]) ** 2).sum(axis=-1) ** 0.5
                centers_diff_idx = centers_diff.argmin(axis=-1)[0]
                centers_diff_i = centers_diff.min(axis=-1)
                mask_appear = centers_diff_i < self.motion_thresh
                indices_appear = np.nonzero(mask_appear)[0]

                for center_idx in range(len(self._centers_each_frame[i])):
                    if center_idx in indices_appear:
                        self._centers_info_list_each_frame[i][center_idx].\
                            center_idx_in_following_frames.append(
                            centers_diff_idx[center_idx]
                        )
                    else:
                        self._centers_info_list_each_frame[i][center_idx].\
                            center_idx_in_following_frames.append(
                            -1
                        )

                # Ideally N bits are enough, but for safety we add a margin 1 bit
                if self.timestamps[j] - self.timestamps[i] >= 2 * self.cycle_time * (self.all_bits + 1):
                    break

    def get_LEDs(self):
        """
        Find all valid bit sequence for each interval and each center
        :return: None
        """
        for i in range(len(self.images)):
            for center_idx in range(len(self._centers_info_list_each_frame[i])):
                info = self._centers_info_list_each_frame[i][center_idx]
                raw_bit_list = [1]
                frame_idx_list = [i]
                timestamp_list = [self.timestamps[i]]
                for frame_delta_idx, center_idx in enumerate(info.center_idx_in_following_frames):
                    raw_bit_list.append(1 if center_idx != -1 else 0)
                    j = i + frame_delta_idx + 1
                    frame_idx_list.append(j)
                    timestamp_list.append(self.timestamps[j])
                if timestamp_list[-1] - timestamp_list[0] < (self.cycle_time * 2 * self.all_bits - 1):
                    # filter intervals which are too short
                    continue
                raw_bit_num = len(raw_bit_list)

                # find sample bits in each cycle
                num_cycles = 1
                cycle_bits_list = []
                raw_bit_idx = 0
                while raw_bit_idx < raw_bit_num:
                    if timestamp_list[raw_bit_idx] - timestamp_list[0] > self.cycle_time * num_cycles:
                        cycle_bits_list.append(raw_bit_list[raw_bit_idx])
                        num_cycles += 1
                    raw_bit_idx = raw_bit_idx + 1
                assert num_cycles >= self.all_bits * 2

                # find manchester encoding bits
                manchester_bit_list = []
                is_valid = True
                for cycle_idx, bit_list in enumerate(cycle_bits_list):
                    if cycle_idx in (0, 1):
                        if sum(bit_list) != len(bit_list):
                            # the samples in the first two cycles must all be '1'
                            is_valid = False
                    elif cycle_idx == 2:
                        # the pattern in the third cycle is '1111...000000'
                        if self.get_diff_bit(bit_list) == 1:
                            is_valid = False
                    else:
                        diff_bit = self.get_diff_bit(bit_list)
                        if diff_bit is not None:
                            manchester_bit_list.append(diff_bit)
                        else:
                            is_valid = False
                    if not is_valid:
                        break

                if not is_valid:
                    continue

                # find differential manchester encoding bits
                manchester_bit_list = manchester_bit_list[:self.all_bits - self.sync_bits]
                num_real_bits = len(manchester_bit_list)
                diff_manchester_bit_list = [sum(manchester_bit_list[:idx+1]) % 2 for idx in range(num_real_bits)]
                real_bit_str = str()
                for bit in diff_manchester_bit_list:
                    real_bit_str += str(bit)
                led_index = int(real_bit_str, 2)
                info.led_index = led_index
                print(f'Find valid interval for LED {led_index} in frame {i}, center {center_idx}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recorder')
    parser.add_argument('--img_dir', type=str, help='the directory of source images')
    args = parser.parse_args()

    img_folder = args.img_dir
    expelled_pixels = [[0, 0]]
    filenames = os.listdir(img_folder)
    filenames.sort()
    images = []
    timestamps = []
    for filename in filenames:
        path = os.path.join(img_folder, filename)
        images.append(cv2.imread(path))

        ts = float(filename.split('_')[-1][:-4])
        timestamps.append(ts)

    tracker = LED_Tracker(images, timestamps, expelled_pixels)
