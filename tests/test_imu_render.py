import cv2
import numpy as np

import pyvista
import tqdm
import transforms3d.euler

import os.path as osp

def drawAxes(rvecs):
    tvecs = np.array([0, 0, 0.2])
    width = 800

    im = np.zeros((width, width, 3))
    cx, cy = width // 2, width // 2
    f = width / 2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float)
    dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    # set the axes size as a function of the marker's
    # square size and project onto image space
    s = 2
    axis = np.float32([[s, 0, 0], [0, s, 0], [0, 0, s]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(
        axis,
        rvecs,
        tvecs,
        K,
        dist
    )

    cv2.drawFrameAxes(
        im,
        K,
        dist,
        rvecs, tvecs,
        length=.1, thickness=5)

    # cv2.drawFrameAxes(
    #     im,
    #     K,
    #     dist,
    #     np.array([-1.14670758, -2.15978896, 0.93563453]), tvecs + np.array([0.2, 0.2, 0.2]),
    #     length=.1, thickness=5)

    return im


#
# def draw_axis(img, rotation_vec, t, K, scale=0.1, dist=None):
#     """
#     Draw a 6dof axis (XYZ -> RGB) in the given rotation and translation
#     :param img - rgb numpy array
#     :rotation_vec - euler rotations, numpy array of length 3,
#                     use cv2.Rodrigues(R)[0] to convert from rotation matrix
#     :t - 3d translation vector, in meters (dtype must be float)
#     :K - intrinsic calibration matrix , 3x3
#     :scale - factor to control the axis lengths
#     :dist - optional distortion coefficients, numpy array of length 4. If None distortion is ignored.
#     """
#     img = img.astype(np.float32)
#     dist = np.zeros(4, dtype=float) if dist is None else dist
#     points = scale * np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
#     axis_points, _ = cv2.projectPoints(points, rotation_vec, t, K, dist)
#     img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[0].ravel()), (255, 0, 0), 3)
#     img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[1].ravel()), (0, 255, 0), 3)
#     img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[2].ravel()), (0, 0, 255), 3)
#     return img


# def correct_offset(x, rotation):
#     mat = transforms3d.euler.euler2mat(*rotation) @ transforms3d.euler.euler2mat(*x)
#     return transforms3d.euler.mat2euler(mat)
#
#
# def calculate_offset(origin, reference):
#     return transforms3d.euler.mat2euler(transforms3d.euler.euler2mat(*reference) @ np.linalg.inv(transforms3d.euler.euler2mat(*origin)))
#
#
# INIT_REFERENCE = np.array([-1.14670758, -2.15978896, 0.93563453])
# INIT_ORIGIN = np.array([-0.7598929266382083, 0.5631959206383874, 1.371692219330263])
# global_fix = calculate_offset(INIT_ORIGIN, INIT_REFERENCE)
#
def create_mesh(quat):
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=float)

    mat = transforms3d.euler.quat2mat(quat)
    nodes[1:4, :] = mat
    edges = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
    ])

    padding = np.empty(edges.shape[0], int) * 2
    padding[:] = 2
    edges_w_padding = np.vstack((padding, edges.T)).T

    return pyvista.PolyData(nodes, edges_w_padding)


def run_once(path_to_imu_data, output_path):
    x = np.load(path_to_imu_data)

    with tqdm.tqdm(total=len(x['roll'])/5) as pbar:
        for i in range(0, len(x['roll']), 5):
            # p = pyvista.Plotter()
            mesh = create_mesh([float(x['quat_w'][i]), float(x['quat_x'][i]), float(x['quat_y'][i]), float(x['quat_z'][i])])
            cpos, img = mesh.plot(
                scalars=["red", "green", "blue"],
                render_lines_as_tubes=True,
                cpos=pyvista.CameraPosition(focal_point=[0, 0, 0], position=[5, -10, 2.5], viewup=[0, 0, 1]),
                style='wireframe',
                line_width=10,
                cmap='jet',
                show_scalar_bar=False,
                background=None,
                off_screen=True,
                return_img=True,
                return_cpos=True,
                screenshot=True,
            )
            # p.add_mesh(mesh, scalars=["red", "green", "blue"],cmap='jet',render_lines_as_tubes=True, style='wireframe', line_width=10, show_scalar_bar=False,)
            # cpos, img = p.show(interactive=False, return_img=True, return_cpos=True) #  background=None,off_screen=False,
            # print(img)
            cv2.imwrite(osp.join(output_path, "frame_{:05d}.png".format(i)), img)
            pbar.update()
            # cv2.imshow('img', img)

            key = cv2.waitKey(1)

            if key == 27:  # 按esc键退出
                print('esc break...')
                cv2.destroyAllWindows()
                break


# run_once(r"C:\Users\liyutong\Downloads\video\swing\imu_mem_2023-01-13_200933\imu_30c6f751c60c.npz",r"C:\Users\liyutong\Downloads\video\renders\swing_render")
run_once(r"C:\Users\liyutong\Downloads\video\aruco_compare\scene1\imu\imu_30c6f751c60c.npz",r"C:\Users\liyutong\Downloads\video\renders\aruco_compare_render\scene1")
run_once(r"C:\Users\liyutong\Downloads\video\aruco_compare\scene2\imu\imu_30c6f751c60c.npz",r"C:\Users\liyutong\Downloads\video\renders\aruco_compare_render\scene2")
run_once(r"C:\Users\liyutong\Downloads\video\aruco_compare\scene3\imu\imu_30c6f751c60c.npz",r"C:\Users\liyutong\Downloads\video\renders\aruco_compare_render\scene3")
run_once(r"C:\Users\liyutong\Downloads\video\figure_1\imu\imu_30c6f751c60c.npz",r"C:\Users\liyutong\Downloads\video\renders\figure_1_render")

#
# import streamlit as st
#
# path_to_imu_data = r"C:\Users\liyutong\Downloads\video\demo\imu\imu_30c6f751c60c.npz"
# fix_roll = st.slider('fix_roll', -np.pi, np.pi, 0.)
# fix_pitch = st.slider('fix_pitch', -np.pi, np.pi, 0.)
# fix_yaw = st.slider('fix_yaw', -np.pi, np.pi, 0.)
# x = np.load(path_to_imu_data)
# im = drawAxes(correct_offset(transforms3d.euler.quat2euler([x['quat_w'][0], x['quat_x'][0], x['quat_y'][0], x['quat_z'][0]]), [fix_roll, fix_pitch, fix_yaw]))
# st.image(im / 255)
