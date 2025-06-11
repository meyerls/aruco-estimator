#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
import logging
from typing import Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image


def ray_cast_aruco_corners(
    extrinsics: np.ndarray, intrinsics: np.ndarray, corners: tuple
) -> Tuple[np.ndarray, np.ndarray]:
    """
    n = x @ K^-1 @ R.T

    :param extrinsics:
    :param intrinsics:
    :param corners:
    :return:
    """
    R, camera_origin = extrinsics[:3, :3], extrinsics[:3, 3]
    aruco_corners = np.concatenate((corners[0][0], np.ones((4, 1))), axis=1)
    rays = aruco_corners @ np.linalg.inv(intrinsics).T @ R.T
    rays_norm = rays / np.linalg.norm(rays, ord=2, axis=1, keepdims=True)

    return camera_origin, rays_norm


def detect_aruco_marker(
    image: np.ndarray,
    dict_type: int = cv2.aruco.DICT_4X4_100,
    aruco_parameters: cv2.aruco.DetectorParameters = cv2.aruco.DetectorParameters(),
) -> Tuple[tuple, np.ndarray]:
    """
    Info: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
    More information on aruco parameters: https://docs.opencv.org/4.x/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html

    @param dict_type:
    @param image:
    @param aruco_parameters:

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

    image = cv2.imread(image)
    if image is None:
        logging.warning(f"Failed to load image: {image}")
        return None, None
    image_size = image.shape
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_parameters)
    corners, aruco_id, _ = detector.detectMarkers(image)
    if aruco_id is None:
        return None, None, image_size
    return corners, aruco_id, image_size
