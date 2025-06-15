"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

import logging
from typing import Tuple
import numpy as np
from .opt import kabsch_umeyama

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
        ])

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

def get_normalization_transform(
    aruco_corners_3d: np.ndarray, true_aruco_size: float
) -> np.ndarray:
    """Calculate transformation matrix to normalize coordinates to ArUco marker plane with correct scaling using Kabsch-Umeyama algorithm."""
    if len(aruco_corners_3d) != 4:
        raise ValueError(f"Expected 4 ArUco corners, got {len(aruco_corners_3d)}")

    # Define target corners: a square centered at origin with the desired size
    half_size = true_aruco_size / 2
    target_corners = np.array([
        [-half_size, half_size, 0],    # top-left
        [half_size, half_size, 0],    # top-right
        [half_size, -half_size, 0],   # bottom-right  
        [-half_size, -half_size, 0],  # bottom-left
    ])
    
    # Calculate edge lengths for logging
    edge1_length = np.linalg.norm(aruco_corners_3d[1] - aruco_corners_3d[0])
    edge2_length = np.linalg.norm(aruco_corners_3d[3] - aruco_corners_3d[0]) 
    avg_measured_size = (edge1_length + edge2_length) / 2
    
    logging.info(
        f"ArUco measured width: {edge1_length:.4f}, height: {edge2_length:.4f}"
    )
    
    # Use Kabsch-Umeyama algorithm to find optimal transformation
    # This finds transformation from aruco_corners_3d to target_corners
    R, c, t = kabsch_umeyama(target_corners, aruco_corners_3d)
    
    # Convert to 4x4 transformation matrix
    transform = get_transformation_matrix_4x4(R, c, t)
    
    # Calculate effective scaling factor for logging
    scale_factor = true_aruco_size / avg_measured_size
    logging.info(
        f"Kabsch-Umeyama scaling factor: {c:.4f}, expected scaling: {scale_factor:.4f}"
    )
    logging.info(
        f"Applied transformation to normalize ArUco to size {true_aruco_size}"
    )
    
    return transform

def get_transformation_matrix_4x4(R, c, t):
    """
    Convert Kabsch-Umeyama results to a 4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = c * R
    T[:3, 3] = t
    return T


