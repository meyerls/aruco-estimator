"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

import logging
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


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
    
    # Calculate current ArUco dimensions for logging
    center = np.mean(aruco_corners_3d, axis=0)
    
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

def kabsch_umeyama(pointset_A, pointset_B):
    """
    Kabsch–Umeyama algorithm exactly as described in the source:
    https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
    
    Finds transformation from B to A: A ≈ t + c * R @ B
    """
    assert pointset_A.shape == pointset_B.shape
    n, m = pointset_A.shape

    # Find centroids (μ_A, μ_B in the source)
    EA = np.mean(pointset_A, axis=0)
    EB = np.mean(pointset_B, axis=0)
    
    # Variance of A (σ_A² in the source)
    VarA = np.mean(np.linalg.norm(pointset_A - EA, axis=1) ** 2)

    # Cross-covariance matrix H
    H = ((pointset_A - EA).T @ (pointset_B - EB)) / n
    
    # SVD: H = U @ D @ VT
    U, D, VT = np.linalg.svd(H)
    
    # Detect reflection
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    # Rotation matrix
    R = U @ S @ VT
    
    # Scaling factor - properly handle D (1D array) and S (2D matrix)
    # np.trace(np.diag(D) @ S) = sum of diagonal elements of diag(D) @ S
    # Since both are diagonal matrices, this is sum(D * diag(S))
    S_diag = np.diag(S)  # Extract diagonal: [1, 1, ..., 1, d]
    c = VarA / np.sum(D * S_diag)
    
    # Translation vector
    t = EA - c * R @ EB

    return R, c, t


def get_transformation_matrix_4x4(R, c, t):
    """
    Convert Kabsch-Umeyama results to a 4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = c * R
    T[:3, 3] = t
    return T


def apply_transformation(points, R, c, t):
    """Apply transformation to points: t + c * R @ points"""
    return t + (c * R @ points.T).T


def apply_transformation_4x4(points, T):
    """Apply 4x4 transformation matrix to points"""
    # Convert to homogeneous coordinates
    points_homo = np.column_stack([points, np.ones(points.shape[0])])
    # Apply transformation
    transformed_homo = (T @ points_homo.T).T
    # Convert back to 3D coordinates
    return transformed_homo[:, :3]

