"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

import numpy as np


def intersect(P0: np.ndarray, N: np.ndarray, solve='pseudo') -> np.ndarray:
    '''
    Least Squares Intersection of Lines: https://silo.tips/download/least-squares-intersection-of-lines#modals

    :param P0:
    :param n:
    :param solve:
    :return:
    '''
    # generate the array of all projectors
    projs = np.eye(N.shape[1]) - N[:, :, np.newaxis] * N[:, np.newaxis]  # I - n*n.T

    # Sum over all K lines to get R
    R = projs.sum(axis=0)

    # Sum over all K lines to get q
    q = (projs @ P0[:, :, np.newaxis]).sum(axis=0)

    # Solve LS for Rp = q to find feasible p
    if solve == 'ls':
        p = np.linalg.lstsq(R, q, rcond=None)[0]
    # P_hat = R_pseudo @ q
    elif solve == 'pseudo':
        p = np.linalg.pinv(R) @ q
    else:
        return NotImplementedError

    return p


def ls_intersection_of_lines(P0: np.ndarray, N: np.ndarray) -> np.ndarray:
    p1 = intersect(P0, N[:, 0])
    p2 = intersect(P0, N[:, 1])
    p3 = intersect(P0, N[:, 2])
    p4 = intersect(P0, N[:, 3])

    intersctions_3d = np.concatenate([p1, p2, p3, p4], axis=1).T

    return intersctions_3d


def intersect_parallelized(P0: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Here the intersection of the rays are calculated. More info can be found in the method directly.
    The original resource for the underlying LS can be found here:
    https://silo.tips/download/least-squares-intersection-of-lines (Sorry this website is quite shady)

    :param P0:
    :param n:
    :return:
    """
    # generate the array of all projectors. Shape (#Corners, #Lines, Dim, Dim)
    I = np.zeros((N.shape[1], N.shape[0], N.shape[2], N.shape[2])) + np.eye(N.shape[2])
    # I - n*n.T
    projs = I - N.transpose(1, 0, 2)[..., np.newaxis] * N.transpose(1, 0, 2)[:, :, np.newaxis]

    # Sum over all K lines to get R
    R = projs.sum(axis=1)

    # Sum over all K lines to get q
    q = (projs @ P0[:, :, np.newaxis]).sum(axis=1)

    # P_hat = R_pseudo @ q
    p = np.linalg.pinv(R) @ q

    return p[..., 0]


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
