"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

import logging
from copy import copy

import numpy as np
import open3d as o3d
import logging
from typing import Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )



def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

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

def kabsch_umeyama(pointset_A, pointset_B):
    """
    Kabsch–Umeyama algorithm is a method for aligning and comparing the similarity between two sets of points.
    It finds the optimal translation, rotation and scaling by minimizing the root-mean-square deviation (RMSD)
    of the point pairs.

    Source and Explenation: https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/

    @param pointset_A: array of a set of points in n-dim
    @param pointset_B: array of a set of points in n-dim
    @return: Rotation Matrix (3x3), scaling (scalar) translation vector (3x1)
    """
    assert pointset_A.shape == pointset_B.shape
    n, m = pointset_A.shape

    # Find centroids of both point sets
    EA = np.mean(pointset_A, axis=0)
    EB = np.mean(pointset_B, axis=0)

    VarA = np.mean(np.linalg.norm(pointset_A - EA, axis=1) ** 2)

    # Covariance matrix
    H = ((pointset_A - EA).T @ (pointset_B - EB)) / n

    # SVD H = UDV^T
    U, D, VT = np.linalg.svd(H)

    # Detect and prevent reflection
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    # rotation, scaling and translation
    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t


def align_point_set(point_set_A, point_set_B):
    R, c, t = kabsch_umeyama(np.asarray(point_set_A), np.asarray(point_set_B))

    point_set_B = np.array([t + c * R @ b for b in point_set_B])

    return point_set_A, point_set_B, [R, c, t]


def plot_aligned_pointset(A, B):
    """
    Visualize transformed point set
    @param A: array of a set of points in n-dim
    @param B: array of a set of points in n-dim
    @return: both point clouds
    """


    pcdA = o3d.geometry.PointCloud()
    pcdA.points = o3d.utility.Vector3dVector(A)

    pcdB = o3d.geometry.PointCloud()
    pcdB.points = o3d.utility.Vector3dVector(B)


    o3d.visualization.draw_geometries([pcdA, pcdB])

    return pcdA, pcdB


def get_icp_transformation(source, target, trafo, max_iteration=2000):
    threshold = 0.02
    trans_init = np.eye(4)
    trans_init[:3, :4] = np.hstack([trafo[1] * trafo[0], np.expand_dims(trafo[2], axis=0).T])

    logging.info("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    logging.info(evaluation)

    logging.info("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

    return reg_p2p

