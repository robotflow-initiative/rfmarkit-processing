from typing import List, Tuple
import cv2
import os
import numpy as np
import json

extrinsic_folder_path = './data/saved_led/saved'
cam_serials = ['001622071104', '011422071122']

img_folder_path = './data/saved_led_blink/saved'
filenames = {
    '001622071104': '000005_1636467722.2903671.jpg',
    '011422071122': '000005_1636467722.2963512.jpg',
}


class get_click_point_callback:
    def __init__(self):
        self.point = None

    def __call__(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f'current point:\t({x}, {y})')
            self.point = (x, y)
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def get_selected_pixel(img):
    click_event = get_click_point_callback()

    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return click_event.point


def get_proj_mat(extrinsic: np.array, intrinsic: np.array):
    intr3x4 = np.zeros((3, 4))
    intr3x4[:3, :3] = intrinsic
    return intr3x4 @ extrinsic


def get_multi_view_equation(proj_mats: List[np.array], pixels: List[float]):
    rows = []
    for p, pixel in zip(proj_mats, pixels):
        rows.append(pixel[0] * p[2].T - p[0].T)
        rows.append(pixel[1] * p[2].T - p[1].T)
    homo_mat = np.stack(rows, axis=0)
    return homo_mat[:, :3], -homo_mat[:, -1]


if __name__ == '__main__':
    # load 4x4 extrinsics
    extrinsics = {}
    for cam in cam_serials:
        extrinsic_path = os.path.join(extrinsic_folder_path, cam, 'extrinsic.npy')
        extrinsics[cam] = np.load(extrinsic_path)

    intrinsics = {}
    # load 3x3 intrinsics
    for cam in cam_serials:
        intrinsics_path = os.path.join(extrinsic_folder_path, cam, 'camera_intrinsic.json')
        with open(intrinsics_path, 'rb') as f:
            intrinsics[cam] = np.array(json.load(f)['intrinsic_matrix'], dtype=np.float).reshape(3, 3).T

    # loaded images are for point annotation and visualization 
    imgs = {}
    coords = {}
    for cam, filename in filenames.items():
        img_path = os.path.join(img_folder_path, cam, 'color', filename)
        imgs[cam] = cv2.imread(img_path)
        print(f'{cam} img shape: {imgs[cam].shape}')

    for cam, img in imgs.items():
        coord = get_selected_pixel(img)
        while coord is None:
            print('no selected point!')
            coord = get_selected_pixel(img)
        coords[cam] = coord
        print(f'selected point:\t {coord}')

    # get projection matrices 
    proj_mats = []
    pixels = []
    for cam in cam_serials:
        proj_mats.append(get_proj_mat(extrinsics[cam], intrinsics[cam]))
        pixels.append(coords[cam])

    # calculate the solution
    A, b = get_multi_view_equation(proj_mats, pixels)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    print(f'final solution: {x}')
