#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
from typing import Union
from typing import Tuple

# Libs
import cv2
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt
from PIL import Image

# Own modules
try:
    from .visualization import *
except ImportError:
    from visualization import *


def ray_cast_aruco_corners_visualization(extrinsics: np.ndarray, intrinsics: np.ndarray, corners: tuple,
                                         corners3d: np.ndarray) \
        -> o3d.pybind.geometry.LineSet:
    """

    :param extrinsics:
    :param intrinsics:
    :param corners:
    :return:
    """
    if corners == None:
        return generate_line_set(points=[],
                                 lines=[],
                                 color=[1, 0, 0])

    p0, rays_norm = ray_cast_aruco_corners(extrinsics, intrinsics, corners)

    p1_0, p1_1, p1_2, p1_3 = corners3d[0], corners3d[1], corners3d[2], corners3d[3]
    t_0 = np.mean((p1_0 - p0) / rays_norm[0])
    t_1 = np.mean((p1_1 - p0) / rays_norm[1])
    t_2 = np.mean((p1_2 - p0) / rays_norm[2])
    t_3 = np.mean((p1_3 - p0) / rays_norm[3])

    points_camera_plane = [
        p0,
        p0 + rays_norm[0] * t_0,  # p1_0,
        p0 + rays_norm[1] * t_1,  # p1_1,
        p0 + rays_norm[2] * t_2,  # p1_2,
        p0 + rays_norm[3] * t_3,  # p1_3,
    ]

    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    aruco_line_set = generate_line_set(points=points_camera_plane,
                                       lines=lines,
                                       color=[1, 0, 0])

    return aruco_line_set


def ray_cast_aruco_corners(extrinsics: np.ndarray, intrinsics: np.ndarray, corners: tuple) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''

    n = x @ K^-1 @ R.T

    :param extrinsics:
    :param intrinsics:
    :param corners:
    :return:
    '''
    R, p0 = extrinsics[:3, :3], extrinsics[:3, 3]
    aruco_corners = np.concatenate((corners[0][0], np.ones((4, 1))), axis=1)
    rays = aruco_corners @ np.linalg.inv(intrinsics).T @ R.T
    rays_norm = rays / np.linalg.norm(rays, ord=2, axis=1, keepdims=True)

    return p0, rays_norm


def load_image(image_path: str) -> np.ndarray:
    """
    Load Image. This takes almost 50% of the time. Would be nice if it is possible to speed up this process. Any
    ideas?

    :param image_path:
    :return:
    """
    return np.asarray(Image.open(image_path))


class ArucoDetection:
    def __init__(self, dict_type: int = aruco.DICT_4X4_1000):
        """
        More information on aruco parameters: https://docs.opencv.org/4.x/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html

        @param dict_type:
        """
        self.dict_type = dict_type
        self.aruco_dict = aruco.Dictionary_get(dict_type)
        self.aruco_parameters = aruco.DetectorParameters_create()

        # aruco_parameters = aruco.DetectorParameters_create()
        # aruco_parameters.adaptiveThreshConstant = 9.0
        # aruco_parameters.adaptiveThreshWinSizeMax = 369
        # aruco_parameters.adaptiveThreshWinSizeMin = 7
        # aruco_parameters.adaptiveThreshWinSizeStep = 49
        # aruco_parameters.cornerRefinementWinSize = 9
        # aruco_parameters.minDistanceToBorder = 7
        # aruco_parameters.cornerRefinementMaxIterations = 149
        # aruco_parameters.minOtsuStdDev = 4.0
        #
        # aruco_parameters.minMarkerDistanceRate = 0.05
        # aruco_parameters.minMarkerPerimeterRate = 5
        # aruco_parameters.maxMarkerPerimeterRate = 10
        #
        #
        # aruco_parameters.polygonalApproxAccuracyRate = 0.05
        # aruco_parameters.minCornerDistanceRate = 0.05

    def detect_aruco_marker(self, image: Union[np.ndarray, str]) -> Tuple[tuple, np.ndarray, np.ndarray]:
        return detect_aruco_marker(image=image, dict_type=self.aruco_dict, aruco_parameters=self.aruco_parameters)


def detect_aruco_marker(image: np.ndarray, dict_type: int = aruco.DICT_4X4_1000,
                        aruco_parameters: cv2.aruco.DetectorParameters = aruco.DetectorParameters_create()) -> Tuple[
    tuple, np.ndarray, np.ndarray]:
    """
    More information on aruco parameters: https://docs.opencv.org/4.x/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html

    @param dict_type:
    @param image:
    @param aruco_parameters:
    """

    # Info: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
    aruco_dict = aruco.Dictionary_get(dict_type)
    aruco_parameters = aruco.DetectorParameters_create()

    aruco_parameters.polygonalApproxAccuracyRate = 0.01
    aruco_parameters.minMarkerPerimeterRate = 0.1
    aruco_parameters.maxMarkerPerimeterRate = 4.0
    """
    Aruco Corners

        p1------------p2
        |             |
        |             |
        |             |
        |             |
        p4------------p3

    :param image:
    :return:
    """
    if isinstance(image, np.ndarray):
        pass
    elif isinstance(image, str):
        image = load_image(image_path=image)
    else:
        raise NotImplementedError

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, aruco_id, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                                 parameters=aruco_parameters)

    if aruco_id is None:
        return None, None, cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if True:
        plt.figure()
        plt.imshow(gray, cmap='gray')
        for i in range(len(aruco_id)):
            c = corners[i][0]
            plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label="id={0}".format(aruco_id[i]))
        plt.legend()
        plt.show()

    return corners, aruco_id, image
