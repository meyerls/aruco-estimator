#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
from typing import Tuple

# Libs
import numpy as np
import open3d as o3d
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion

# Own modules
from colmap_wrapper.visualization import *


def ray_cast_aruco_corners_visualization(p_i: np.ndarray, n_i: np.ndarray, corners3d: np.ndarray) \
        -> o3d.pybind.geometry.LineSet:
    '''

    @param p_i:
    @param n_i:
    @param corners3d:
    @return:
    '''

    p1_0, p1_1, p1_2, p1_3 = corners3d[0], corners3d[1], corners3d[2], corners3d[3]
    t_0 = np.linalg.norm((p1_0 - p_i))
    t_1 = np.linalg.norm((p1_1 - p_i))
    t_2 = np.linalg.norm((p1_2 - p_i))
    t_3 = np.linalg.norm((p1_3 - p_i))

    points_camera_plane = [
        p_i,
        p_i + n_i[0] * t_0,  # p1_0,
        p_i + n_i[1] * t_1,  # p1_1,
        p_i + n_i[2] * t_2,  # p1_2,
        p_i + n_i[3] * t_3,  # p1_3,
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


