"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

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
    
def get_corners_at_origin(side_length=1):
    # Define target corners: a square centered at origin with the desired size
    half_size = side_length / 2
    target_corners = np.array([
        [-half_size, half_size, 0],    # top-left
        [half_size, half_size, 0],    # top-right
        [half_size, -half_size, 0],   # bottom-right  
        [-half_size, -half_size, 0],  # bottom-left
    ])
    return target_corners

def get_transformation_between_clouds(
    aruco_corners_3d: np.ndarray, target_corners: np.ndarray
) -> np.ndarray:
    """Calculate transformation matrix to normalize coordinates to ArUco marker plane with correct scaling using Kabsch-Umeyama algorithm."""
    if len(aruco_corners_3d[0]) != 3:
        raise ValueError(f"Expected 3D ArUco corners, got {aruco_corners_3d}")

    # Use Kabsch-Umeyama algorithm to find optimal transformation
    # This finds transformation from aruco_corners_3d to target_corners
    R, c, t = kabsch_umeyama(target_corners, aruco_corners_3d)
    
    # Convert to 4x4 transformation matrix
    transform = get_transformation_matrix_4x4(R, c, t)
 
    return transform

def get_transformation_matrix_4x4(R, c, t):
    """
    Convert Kabsch-Umeyama results to a 4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = c * R
    T[:3, 3] = t
    return T


