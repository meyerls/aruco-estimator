#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Memory-optimized marker detection with minimal peak memory usage and distortion support.
Processes one marker at a time and uses streaming/generator approaches.
"""

import logging
import os
from enum import Enum
from typing import Dict, Tuple, List, Optional, Union, Generator, Iterator
from collections import defaultdict
from dataclasses import dataclass
import gc

import cv2
import numpy as np
from tqdm import tqdm
from pupil_apriltags import Detector as AprilTagDetector

# Import your line/line intersection routines
from ..opt import intersect_parallelized, intersect


# ---------------------------------------------------------------------
# Memory-efficient data structures
# ---------------------------------------------------------------------


@dataclass
class RayData:
    """Lightweight ray data structure"""

    P0: np.ndarray  # (3,) camera center
    N: np.ndarray  # (4,3) ray directions for 4 corners
    image_id: int

    def __post_init__(self):
        # Ensure arrays are compact
        self.P0 = np.ascontiguousarray(self.P0, dtype=np.float32)
        self.N = np.ascontiguousarray(self.N, dtype=np.float32)


class MarkerType(Enum):
    ARUCO = "aruco"
    APRILTAG = "apriltag"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def _safe_load_image(project, image_id: int) -> Optional[np.ndarray]:
    """Safely load an image with error handling."""
    try:
        return project.load_image_by_id(image_id)
    except Exception as e:
        logging.warning(f"Failed to load image {image_id}: {e}")
        return None


def _safe_convert_to_gray(image: np.ndarray) -> Optional[np.ndarray]:
    """Safely convert image to grayscale."""
    try:
        if image is None:
            return None
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image
    except Exception as e:
        logging.warning(f"Failed to convert image to grayscale: {e}")
        return None


def _undistort_corners_if_needed(
    corners: np.ndarray, camera, undistort_points: bool = True
) -> np.ndarray:
    """
    Undistort corner points using camera model if requested.

    Args:
        corners: Nx2 array of corner points
        camera: Camera object with K and D properties
        undistort_points: Whether to apply undistortion

    Returns:
        Undistorted corner points (or original if undistort_points=False)
    """
    if not undistort_points:
        return corners

    try:
        # Check if we have distortion to correct
        if hasattr(camera, "D") and np.any(camera.D != 0):
            if camera.is_fisheye() if hasattr(camera, "is_fisheye") else False:
                # Use fisheye undistortion
                undistorted = cv2.fisheye.undistortPoints(
                    corners.reshape(-1, 1, 2),
                    camera.K,
                    camera.D,
                    P=camera.K,
                )
            else:
                # Use standard OpenCV undistortion
                undistorted = cv2.undistortPoints(
                    corners.reshape(-1, 1, 2),
                    camera.K,
                    camera.D,
                    P=camera.K,
                )
            return undistorted.reshape(-1, 2)
        else:
            # No distortion to correct
            return corners
    except Exception as e:
        logging.warning(f"Failed to undistort points: {e}, using original points")
        return corners


# ---------------------------------------------------------------------
# Single-pass detection functions
# ---------------------------------------------------------------------


def _detect_all_markers_single_pass_apriltag(
    project,
    tag_family: str,
    progress_bar: bool = True,
    undistort_points: bool = True,
) -> Dict[int, List[RayData]]:
    """
    Single-pass detection that returns all ray data for all markers with distortion support.
    Much more efficient than multi-pass approaches.
    """
    image_ids = list(project.images.keys())

    # Create detector once and reuse
    try:
        detector = AprilTagDetector(families=tag_family)
        logging.info(f"Created AprilTag detector for family: {tag_family}")
    except Exception as e:
        logging.error(f"Failed to create AprilTag detector: {e}")
        return {}

    # Single pass: collect all ray data organized by marker ID
    marker_ray_data = {}  # marker_id -> list of RayData

    pbar = tqdm(
        total=len(image_ids), disable=not progress_bar, desc="Detecting AprilTags"
    )
    for image_id in image_ids:
        try:
            image = project.load_image_by_id(image_id)
            if image is None:
                pbar.update(1)
                continue

            h, w = image.shape[:2]
            gray = _safe_convert_to_gray(image)
            if gray is None:
                pbar.update(1)
                continue

            results = detector.detect(gray)

            if results:  # Process all detected markers in this image
                image_meta = project.images[image_id]
                camera = project.cameras[image_meta.camera_id]
                sx = camera.width / w
                sy = camera.height / h

                for r in results:
                    marker_id = int(r.tag_id)

                    # Reorder corners: bl,br,tr,tl -> tl,tr,br,bl
                    pts = np.array(
                        [r.corners[3], r.corners[2], r.corners[1], r.corners[0]]
                        # r.corners
                    )
                    scaled = pts.copy()
                    scaled[:, 0] *= sx
                    scaled[:, 1] *= sy

                    # Apply distortion correction if requested
                    if undistort_points:
                        scaled = _undistort_corners_if_needed(
                            scaled, camera, undistort_points
                        )

                    p0, n = ray_cast_marker_corners(
                        extrinsics=image_meta.world_extrinsics,
                        intrinsics=camera.K,
                        corners=scaled,
                    )

                    ray_data = RayData(P0=p0, N=n, image_id=image_id)

                    if marker_id not in marker_ray_data:
                        marker_ray_data[marker_id] = []
                    marker_ray_data[marker_id].append(ray_data)

            # Clean up image data immediately
            del image, gray

        except Exception as e:
            logging.warning(f"Error processing image {image_id}: {e}")

        pbar.update(1)

    pbar.close()

    logging.info(
        f"AprilTag single-pass detection found {len(marker_ray_data)} unique markers"
    )
    return marker_ray_data


def _detect_all_markers_single_pass_aruco(
    project,
    aruco_dict,
    detector_params,
    progress_bar: bool = True,
    undistort_points: bool = True,
) -> Dict[int, List[RayData]]:
    """
    Single-pass ArUco detection that returns all ray data for all markers with distortion support.
    """
    image_ids = list(project.images.keys())
    marker_ray_data = {}  # marker_id -> list of RayData

    pbar = tqdm(
        total=len(image_ids), disable=not progress_bar, desc="Detecting ArUco markers"
    )

    for image_id in image_ids:
        try:
            image = _safe_load_image(project, image_id)
            if image is None:
                pbar.update(1)
                continue

            h, w = image.shape[:2]

            detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
            corners, marker_ids, _ = detector.detectMarkers(image)

            if corners is not None and marker_ids is not None:
                image_meta = project.images[image_id]
                camera = project.cameras[image_meta.camera_id]
                sx = camera.width / w
                sy = camera.height / h

                for corner_set, marker_id in zip(corners, marker_ids.flatten()):
                    marker_id = int(marker_id)
                    pts = corner_set[0]  # (4,2)
                    scaled = pts.copy()
                    scaled[:, 0] *= sx
                    scaled[:, 1] *= sy

                    # Apply distortion correction if requested
                    if undistort_points:
                        scaled = _undistort_corners_if_needed(
                            scaled, camera, undistort_points
                        )

                    p0, n = ray_cast_marker_corners(
                        extrinsics=image_meta.world_extrinsics,
                        intrinsics=camera.K,
                        corners=scaled,
                    )

                    ray_data = RayData(P0=p0, N=n, image_id=image_id)

                    if marker_id not in marker_ray_data:
                        marker_ray_data[marker_id] = []
                    marker_ray_data[marker_id].append(ray_data)

            # # Clean up immediately
            # del image, detector, corners, marker_ids

        except Exception as e:
            logging.warning(f"Error processing image {image_id}: {e}")

        pbar.update(1)

    pbar.close()
    logging.info(
        f"ArUco single-pass detection found {len(marker_ray_data)} unique markers"
    )
    return marker_ray_data


# ---------------------------------------------------------------------
# Memory-efficient detection generators (kept for compatibility)
# ---------------------------------------------------------------------


def _detect_markers_streaming_aruco(
    project,
    aruco_dict,
    detector_params,
    progress_bar: bool = True,
    undistort_points: bool = True,
) -> Iterator[Tuple[int, List[RayData]]]:
    """
    Generator that yields (marker_id, ray_data_list) one marker at a time.
    Now uses single-pass detection internally for efficiency.
    """
    # Get all marker data in single pass
    all_marker_data = _detect_all_markers_single_pass_aruco(
        project, aruco_dict, detector_params, progress_bar, undistort_points
    )

    # Yield one marker at a time
    for marker_id in sorted(all_marker_data.keys()):
        yield marker_id, all_marker_data[marker_id]
        # Clean up after yielding
        del all_marker_data[marker_id]
        gc.collect()


def _detect_markers_streaming_apriltag(
    project,
    tag_family: str,
    progress_bar: bool = True,
    undistort_points: bool = True,
) -> Iterator[Tuple[int, List[RayData]]]:
    """
    Generator that yields (marker_id, ray_data_list) one marker at a time for AprilTags.
    Now uses single-pass detection internally for efficiency.
    """
    # Get all marker data in single pass
    all_marker_data = _detect_all_markers_single_pass_apriltag(
        project, tag_family, progress_bar, undistort_points
    )

    # Yield one marker at a time
    for marker_id in sorted(all_marker_data.keys()):
        yield marker_id, all_marker_data[marker_id]
        # Clean up after yielding
        del all_marker_data[marker_id]
        gc.collect()


# ---------------------------------------------------------------------
# Optimized 3D position calculation
# ---------------------------------------------------------------------


def _calculate_3d_position_single_marker(
    ray_data_list: List[RayData],
    marker_id: int,
    marker_type: MarkerType,
    min_detections: int,
    ransac_config: Dict,
) -> Optional[Dict]:
    """
    Calculate 3D position for a single marker from its ray data.
    Uses memory-efficient processing.
    """
    if len(ray_data_list) < min_detections:
        logging.debug(
            f"{marker_type.value} ID {marker_id}: Only {len(ray_data_list)} detections, need {min_detections}"
        )
        return None

    # Convert to arrays efficiently
    n_detections = len(ray_data_list)
    P0_array = np.empty((n_detections, 3), dtype=np.float32)
    N_array = np.empty((n_detections, 4, 3), dtype=np.float32)
    image_ids = []
    corner_pixels = []  # Store 2D pixel coordinates for each detection

    for i, ray_data in enumerate(ray_data_list):
        P0_array[i] = ray_data.P0
        N_array[i] = ray_data.N
        image_ids.append(ray_data.image_id)

    corners_3d = []
    corner_inlier_counts = []

    # Process each corner
    for corner_idx in range(4):
        centers = P0_array  # (n,3)
        dirs = N_array[:, corner_idx, :]  # (n,3)

        pt3d, inliers = ransac_ray_intersection(
            centers,
            dirs,
            max_iterations=ransac_config["max_iterations"],
            distance_threshold=ransac_config["distance_threshold"],
            min_inliers=ransac_config["min_inliers"],
        )

        if pt3d is None:
            logging.warning(
                f"{marker_type.value} ID {marker_id}, Corner {corner_idx}: RANSAC failed"
            )
            continue

        corners_3d.append(pt3d)
        corner_inlier_counts.append(int(inliers.size))

    if len(corners_3d) == 4:
        corners_3d = np.array(corners_3d, dtype=np.float32)
        center_xyz = np.mean(corners_3d, axis=0)
        quality = float(np.mean(corner_inlier_counts)) / float(n_detections)

        result = {
            "corners_3d": corners_3d,
            "center_xyz": center_xyz,
            "image_ids": image_ids,
            "corner_pixels": corner_pixels,  # Add pixel coordinates
            "marker_type": marker_type.value,
            "total_detections": n_detections,
            "corner_inlier_counts": corner_inlier_counts,
            "detection_quality": quality,
        }

        if quality < 0.9:
            return None

        logging.info(
            f"{marker_type.value} ID {marker_id}: "
            f"detections={n_detections}, "
            f"avg_inliers={np.mean(corner_inlier_counts):.1f}, "
            f"quality={quality:.2f}"
        )

        return result
    else:
        logging.warning(
            f"{marker_type.value} ID {marker_id}: Only {len(corners_3d)}/4 corners localized"
        )
        return None


# ---------------------------------------------------------------------
# Updated public API with distortion support
# ---------------------------------------------------------------------


def localize_aruco_markers(
    project,
    dict_type: int,
    detector: cv2.aruco.ArucoDetector,
    progress_bar: bool = True,
    num_processes: int = None,
    min_detections: int = 3,
    ransac_config: Dict = None,
    undistort_points: bool = True,
) -> Dict[int, Dict]:
    """
    Memory-optimized ArUco marker localization with distortion support.
    Uses single-pass detection for maximum efficiency.

    Args:
        project: SfM project object
        dict_type: ArUco dictionary type
        detector: ArUco detector
        progress_bar: Show progress bar
        num_processes: Number of processes (unused, kept for compatibility)
        min_detections: Minimum detections required per marker
        ransac_config: RANSAC configuration
        undistort_points: Whether to undistort detected corners using camera model
    """
    if ransac_config is None:
        ransac_config = {
            "max_iterations": 2000,
            "distance_threshold": 0.05,
            "min_inliers": 5,
        }

    logging.info(f"Processing ArUco dictionary type: {dict_type}")
    logging.info(f"Minimum detections required: {min_detections}")
    logging.info(f"Undistort points: {undistort_points}")

    detector_params = detector.getDetectorParameters()
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)

    # Single-pass detection to get all marker ray data
    all_marker_data = _detect_all_markers_single_pass_aruco(
        project, aruco_dict, detector_params, progress_bar, undistort_points
    )

    marker_results = {}

    # Process each marker's 3D position
    for marker_id, ray_data_list in all_marker_data.items():
        result = _calculate_3d_position_single_marker(
            ray_data_list=ray_data_list,
            marker_id=marker_id,
            marker_type=MarkerType.ARUCO,
            min_detections=min_detections,
            ransac_config=ransac_config,
        )

        if result is not None:
            marker_results[marker_id] = result

        # Force cleanup after each marker
        del ray_data_list
        gc.collect()

    if marker_results:
        logging.info(f"ArUco: Successfully localized {len(marker_results)} markers")
    else:
        logging.warning("ArUco: No markers successfully localized")

    return marker_results


def localize_apriltag_markers(
    project,
    tag_family: str = "tag36h11",
    progress_bar: bool = True,
    min_detections: int = 3,
    ransac_config: Dict = None,
    undistort_points: bool = True,
) -> Dict[int, Dict]:
    """
    Memory-optimized AprilTag marker localization with distortion support.
    Uses single-pass detection for maximum efficiency.

    Args:
        project: SfM project object
        tag_family: AprilTag family to detect
        progress_bar: Show progress bar
        min_detections: Minimum detections required per marker
        ransac_config: RANSAC configuration
        undistort_points: Whether to undistort detected corners using camera model
    """
    if ransac_config is None:
        ransac_config = {
            "max_iterations": 1000,
            "distance_threshold": 0.1,
            "min_inliers": 3,
        }

    logging.info(f"Processing AprilTag family: {tag_family}")
    logging.info(f"Minimum detections required: {min_detections}")
    logging.info(f"Undistort points: {undistort_points}")

    # Single-pass detection to get all marker ray data
    all_marker_data = _detect_all_markers_single_pass_apriltag(
        project, tag_family, progress_bar, undistort_points
    )

    marker_results = {}

    # Process each marker's 3D position
    for marker_id, ray_data_list in all_marker_data.items():
        result = _calculate_3d_position_single_marker(
            ray_data_list=ray_data_list,
            marker_id=marker_id,
            marker_type=MarkerType.APRILTAG,
            min_detections=min_detections,
            ransac_config=ransac_config,
        )

        if result is not None:
            marker_results[marker_id] = result

        # Force cleanup after each marker
        del ray_data_list
        gc.collect()

    if marker_results:
        logging.info(f"AprilTag: Successfully localized {len(marker_results)} markers")
    else:
        logging.warning("AprilTag: No markers successfully localized")

    return marker_results


# Keep original function for backward compatibility
def localize_markers(
    project,
    dict_type: int,
    detector: cv2.aruco.ArucoDetector,
    progress_bar: bool = True,
    num_processes: int = None,
    min_detections: int = 3,
    ransac_config: Dict = None,
    undistort_points: bool = True,
) -> Dict[int, Dict]:
    """Backward-compatible ArUco entrypoint with distortion support."""
    return localize_aruco_markers(
        project=project,
        dict_type=dict_type,
        detector=detector,
        progress_bar=progress_bar,
        num_processes=num_processes,
        min_detections=min_detections,
        ransac_config=ransac_config,
        undistort_points=undistort_points,
    )


# ---------------------------------------------------------------------
# Existing helper functions (unchanged but included for completeness)
# ---------------------------------------------------------------------


def ransac_ray_intersection(
    camera_centers: np.ndarray,
    ray_directions: np.ndarray,
    max_iterations: int = 1000,
    distance_threshold: float = 0.1,
    min_inliers: int = 3,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """RANSAC ray intersection - unchanged from original."""
    if len(camera_centers) < min_inliers:
        return None, np.array([])

    n_rays = len(camera_centers)
    best_point = None
    best_inliers = np.array([], dtype=int)
    max_inlier_count = 0

    rng = np.random.default_rng()
    for _ in range(max_iterations):
        sample_size = (
            min_inliers
            if n_rays == min_inliers
            else rng.integers(min_inliers, n_rays + 1)
        )
        sample_idx = rng.choice(n_rays, size=sample_size, replace=False)

        P0_s = camera_centers[sample_idx]
        N_s = ray_directions[sample_idx]

        try:
            candidate_point = intersect(P0_s, N_s, solve="pseudo").reshape(-1)
        except (np.linalg.LinAlgError, ValueError):
            continue

        diffs = candidate_point[None, :] - camera_centers
        proj = np.sum(diffs * ray_directions, axis=1)
        closest = camera_centers + proj[:, None] * ray_directions
        dists = np.linalg.norm(candidate_point[None, :] - closest, axis=1)
        inliers = np.where(dists < distance_threshold)[0]

        inlier_count = inliers.size
        if inlier_count >= min_inliers and inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_point = candidate_point.copy()
            best_inliers = inliers

    if best_point is not None and best_inliers.size >= min_inliers:
        try:
            P0_in = camera_centers[best_inliers]
            N_in = ray_directions[best_inliers]
            refined_point = intersect(P0_in, N_in, solve="pseudo").reshape(-1)
            best_point = refined_point
        except (np.linalg.LinAlgError, ValueError):
            pass

    return best_point, best_inliers


def ray_cast_marker_corners(
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    corners: Union[tuple, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Ray casting helper - unchanged from original."""
    R, camera_origin = extrinsics[:3, :3], extrinsics[:3, 3]

    if isinstance(corners, tuple) and len(corners) > 0:
        corner_points = corners[0][0]
    elif isinstance(corners, np.ndarray):
        corner_points = corners
    else:
        raise ValueError(f"Unsupported corner format: {type(corners)}")

    if corner_points.shape != (4, 2):
        raise ValueError(f"Expected corners shape (4, 2), got {corner_points.shape}")

    marker_corners = np.concatenate((corner_points, np.ones((4, 1))), axis=1)
    rays_cam = marker_corners @ np.linalg.inv(intrinsics).T
    rays_world = rays_cam @ R.T

    norms = np.linalg.norm(rays_world, ord=2, axis=1, keepdims=True)
    rays_norm = rays_world / (norms + 1e-12)
    return camera_origin, rays_norm
