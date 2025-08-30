#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced marker detection with support for both ArUco and AprilTag markers.
Includes minimum detection threshold and RANSAC corner estimation.
"""

import logging
import os
import random
from functools import partial
from multiprocessing import Pool
from typing import Dict, Tuple, List, Optional, Union
from enum import Enum

import cv2
import numpy as np
from pupil_apriltags import Detector as AprilTagDetector
from tqdm import tqdm
from ..opt import intersect_parallelized, intersect


class MarkerType(Enum):
    """Supported marker types."""

    ARUCO = "aruco"
    APRILTAG = "apriltag"


def localize_markers(
    project,
    dict_type: int,
    detector: cv2.aruco.ArucoDetector,
    progress_bar: bool = True,
    num_processes: int = None,
    min_detections: int = 3,
    ransac_config: Dict = None,
) -> Dict[int, Dict]:
    """
    Main entry point for ArUco marker detection (for backward compatibility).
    This routes to localize_aruco_markers.

    :param project: SfmProjectBase instance
    :param dict_type: ArUco dictionary type identifier
    :param detector: Configured ArucoDetector
    :param progress_bar: Show progress bars
    :param num_processes: Number of processes for multiprocessing
    :param min_detections: Minimum number of detections required per marker
    :param ransac_config: RANSAC configuration dict
    :return: Dictionary mapping aruco_id -> marker_data
    """
    return localize_aruco_markers(
        project=project,
        dict_type=dict_type,
        detector=detector,
        progress_bar=progress_bar,
        num_processes=num_processes,
        min_detections=min_detections,
        ransac_config=ransac_config,
    )


def ransac_ray_intersection(
    camera_centers: np.ndarray,
    ray_directions: np.ndarray,
    max_iterations: int = 1000,
    distance_threshold: float = 0.1,
    min_inliers: int = 3,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Use RANSAC to robustly estimate 3D point from multiple ray intersections.
    Uses the intersect() function from opts.py for the actual intersection calculation.

    :param camera_centers: Array of camera centers (N, 3)
    :param ray_directions: Array of ray directions (N, 3)
    :param max_iterations: Maximum RANSAC iterations
    :param distance_threshold: Distance threshold for inliers
    :param min_inliers: Minimum number of inliers required
    :return: (best_point_3d, inlier_mask) or (None, empty_mask) if failed
    """
    if len(camera_centers) < min_inliers:
        return None, np.array([])

    n_rays = len(camera_centers)
    best_point = None
    best_inliers = np.array([])
    max_inlier_count = 0

    for iteration in range(max_iterations):
        # Randomly sample rays (at least min_inliers, up to all rays)
        sample_size = min(max(min_inliers, random.randint(min_inliers, n_rays)), n_rays)
        sample_indices = random.sample(range(n_rays), sample_size)

        # Use sampled rays to estimate 3D point using opts.py intersect function
        sample_centers = camera_centers[sample_indices]  # (sample_size, 3)
        sample_directions = ray_directions[sample_indices]  # (sample_size, 3)

        try:
            # Use intersect from opts.py - expects P0: (K, 3), N: (K, 3)
            candidate_point = intersect(
                sample_centers, sample_directions, solve="pseudo"
            )
            candidate_point = candidate_point.flatten()  # Convert to 1D array

        except (np.linalg.LinAlgError, ValueError):
            # Skip this iteration if intersection fails
            continue

        # Test all rays against this candidate point
        inliers = []
        for i in range(n_rays):
            # Calculate distance from ray to candidate point
            ray_to_point = candidate_point - camera_centers[i]

            # Project onto ray direction to get closest point on ray
            projection_length = np.dot(ray_to_point, ray_directions[i])
            closest_point_on_ray = (
                camera_centers[i] + projection_length * ray_directions[i]
            )

            # Distance from candidate point to ray
            distance = np.linalg.norm(candidate_point - closest_point_on_ray)

            if distance < distance_threshold:
                inliers.append(i)

        # Update best solution if this is better
        if len(inliers) > max_inlier_count and len(inliers) >= min_inliers:
            max_inlier_count = len(inliers)
            best_point = candidate_point.copy()
            best_inliers = np.array(inliers)

    # Refine using all inliers with opts.py intersect function
    if best_point is not None and len(best_inliers) >= min_inliers:
        try:
            inlier_centers = camera_centers[best_inliers]
            inlier_directions = ray_directions[best_inliers]

            # Final refinement using all inliers
            refined_point = intersect(inlier_centers, inlier_directions, solve="pseudo")
            best_point = refined_point.flatten()
        except (np.linalg.LinAlgError, ValueError):
            # Keep the original RANSAC point if refinement fails
            pass

    return best_point, best_inliers


def ray_cast_marker_corners(
    extrinsics: np.ndarray, intrinsics: np.ndarray, corners: Union[tuple, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cast rays from camera center through marker corners.

    :param extrinsics: Camera extrinsics matrix
    :param intrinsics: Camera intrinsics matrix
    :param corners: Marker corner coordinates
    :return: (camera_origin, ray_directions)
    """
    R, camera_origin = extrinsics[:3, :3], extrinsics[:3, 3]

    # Handle different corner formats
    if isinstance(corners, tuple) and len(corners) > 0:
        # ArUco format: tuple of arrays
        corner_points = corners[0][0]
    elif isinstance(corners, np.ndarray):
        # AprilTag format: direct numpy array
        corner_points = corners
    else:
        raise ValueError(f"Unsupported corner format: {type(corners)}")

    # Ensure we have the right shape (4, 2)
    if corner_points.shape != (4, 2):
        raise ValueError(f"Expected corners shape (4, 2), got {corner_points.shape}")

    # Convert to homogeneous coordinates
    marker_corners = np.concatenate((corner_points, np.ones((4, 1))), axis=1)
    rays = marker_corners @ np.linalg.inv(intrinsics).T @ R.T
    rays_norm = rays / np.linalg.norm(rays, ord=2, axis=1, keepdims=True)
    return camera_origin, rays_norm


def detect_aruco_markers_in_image(
    image: np.ndarray, dict_type: int, detector_params: cv2.aruco.DetectorParameters
) -> Tuple[Optional[tuple], Optional[np.ndarray], tuple]:
    """
    Detect ArUco markers in a single image.

    :param image: Input image
    :param dict_type: ArUco dictionary type
    :param detector_params: Detector parameters
    :return: (corners, marker_ids, image_size)
    """
    # Create detector inside worker process (for multiprocessing compatibility)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    image_size = image.shape
    corners, marker_ids, _ = detector.detectMarkers(image)
    if marker_ids is None:
        return None, None, image_size
    return corners, marker_ids, image_size


def detect_apriltag_markers_in_image(
    image: np.ndarray, tag_family: str = "tag36h11"
) -> Tuple[Optional[List], Optional[List], tuple]:
    """
    Detect AprilTag markers in a single image using pupil-apriltags.

    :param image: Input image
    :param tag_family: AprilTag family (e.g., "tag36h11")
    :return: (corners, marker_ids, image_size)
    """
    image_size = image.shape

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Create AprilTag detector using pupil-apriltags
    detector = AprilTagDetector(families=tag_family)

    # Detect tags
    results = detector.detect(gray)

    if not results:
        return None, None, image_size

    # Convert to format compatible with ArUco
    corners = []
    marker_ids = []

    for result in results:
        # pupil-apriltags corners are in different order than ArUco
        # pupil-apriltags: bottom-left, bottom-right, top-right, top-left
        # ArUco: top-left, top-right, bottom-right, bottom-left
        # Reorder to match ArUco convention
        corners_reordered = np.array(
            [
                result.corners[3],  # top-left
                result.corners[2],  # top-right
                result.corners[1],  # bottom-right
                result.corners[0],  # bottom-left
            ]
        ).astype(np.float32)

        corners.append(corners_reordered)
        marker_ids.append(result.tag_id)

    return corners, marker_ids, image_size


def localize_aruco_markers(
    project,
    dict_type: int,
    detector: cv2.aruco.ArucoDetector,
    progress_bar: bool = True,
    num_processes: int = None,
    min_detections: int = 3,
    ransac_config: Dict = None,
) -> Dict[int, Dict]:
    """
    Detect and localize ArUco markers in 3D space.

    :param project: SfmProjectBase instance
    :param dict_type: ArUco dictionary type identifier
    :param detector: Configured ArucoDetector
    :param progress_bar: Show progress bars
    :param num_processes: Number of processes for multiprocessing
    :param min_detections: Minimum number of detections required per marker
    :param ransac_config: RANSAC configuration dict
    :return: Dictionary mapping aruco_id -> marker_data
    """
    if num_processes is None:
        num_processes = min(12, os.cpu_count())

    # Disable multiprocessing if only 1 process
    if num_processes <= 1:
        num_processes = None

    # Default RANSAC configuration
    if ransac_config is None:
        ransac_config = {
            "max_iterations": 1000,
            "distance_threshold": 0.1,
            "min_inliers": 3,
        }

    logging.info(f"Processing ArUco dictionary type: {dict_type}")
    logging.info(f"Minimum detections required: {min_detections}")
    logging.info(f"RANSAC config: {ransac_config}")

    # Detect markers in all images
    detection_data = _detect_aruco_markers_in_project(
        project, dict_type, detector, progress_bar, num_processes
    )

    # Calculate 3D positions with minimum detection threshold and RANSAC
    marker_results = _calculate_3d_positions_robust(
        project, MarkerType.ARUCO, detection_data, min_detections, ransac_config
    )

    if marker_results:
        logging.info(f"ArUco: Found {len(marker_results)} markers")
    else:
        logging.warning(f"ArUco: No markers found")

    return marker_results


def localize_apriltag_markers(
    project,
    tag_family: str = "tag36h11",
    progress_bar: bool = True,
    min_detections: int = 3,
    ransac_config: Dict = None,
) -> Dict[int, Dict]:
    """
    Detect and localize AprilTag markers in 3D space using pupil-apriltags.

    :param project: SfmProjectBase instance
    :param tag_family: AprilTag family (e.g., "tag36h11")
    :param progress_bar: Show progress bars
    :param min_detections: Minimum number of detections required per marker
    :param ransac_config: RANSAC configuration dict
    :return: Dictionary mapping tag_id -> marker_data
    """
    # Default RANSAC configuration
    if ransac_config is None:
        ransac_config = {
            "max_iterations": 1000,
            "distance_threshold": 0.1,
            "min_inliers": 3,
        }

    logging.info(f"Processing AprilTag family: {tag_family}")
    logging.info(f"Minimum detections required: {min_detections}")
    logging.info(f"RANSAC config: {ransac_config}")

    # Detect markers in all images
    detection_data = _detect_apriltag_markers_in_project(
        project, tag_family, progress_bar
    )

    # Calculate 3D positions with minimum detection threshold and RANSAC
    marker_results = _calculate_3d_positions_robust(
        project, MarkerType.APRILTAG, detection_data, min_detections, ransac_config
    )

    if marker_results:
        logging.info(f"AprilTag: Found {len(marker_results)} markers")
    else:
        logging.warning(f"AprilTag: No markers found")

    return marker_results


def _detect_aruco_markers_in_project(
    project,
    dict_type: int,
    detector: cv2.aruco.ArucoDetector,
    progress_bar: bool,
    num_processes: int,
) -> Dict:
    """
    Detect ArUco markers in all project images.
    """
    # Load images
    image_ids = list(project._images.keys())
    images = []

    for image_id in tqdm(
        image_ids,
        desc=f"Loading images for ArUco dict {dict_type}",
        disable=not progress_bar,
    ):
        image = project.load_image_by_id(image_id)
        images.append(image)

    # Try multiprocessing if num_processes > 1, fall back to sequential if it fails
    if num_processes and num_processes > 1:
        try:
            # Extract detector parameters for multiprocessing
            detector_params = detector.getDetectorParameters()

            # Detect markers using multiprocessing
            with Pool(num_processes) as p:
                results = list(
                    tqdm(
                        p.imap(
                            partial(
                                detect_aruco_markers_in_image,
                                dict_type=dict_type,
                                detector_params=detector_params,
                            ),
                            images,
                        ),
                        total=len(images),
                        disable=not progress_bar,
                        desc=f"Detecting ArUco (dict {dict_type}) - multiprocessing",
                    )
                )
        except Exception as e:
            logging.warning(
                f"Multiprocessing failed ({e}), falling back to sequential processing"
            )
            results = _detect_aruco_sequential(
                images, detector, dict_type, progress_bar
            )
    else:
        # Sequential processing
        results = _detect_aruco_sequential(images, detector, dict_type, progress_bar)

    return _process_detection_results(project, image_ids, results, MarkerType.ARUCO)


def detect_apriltag_markers_in_image(
    image: np.ndarray, detector: AprilTagDetector
) -> Tuple[Optional[List], Optional[List], tuple]:
    """
    Detect AprilTag markers in a single image using a pre-created detector.

    :param image: Input image
    :param detector: Pre-created AprilTagDetector instance
    :return: (corners, marker_ids, image_size)
    """
    image_size = image.shape

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Detect tags using the passed detector
    try:
        results = detector.detect(gray)
    except Exception as e:
        logging.warning(f"AprilTag detection failed: {e}")
        return None, None, image_size

    if not results:
        return None, None, image_size

    # Convert to format compatible with ArUco
    corners = []
    marker_ids = []

    for result in results:
        # pupil-apriltags corners are in different order than ArUco
        # pupil-apriltags: bottom-left, bottom-right, top-right, top-left
        # ArUco: top-left, top-right, bottom-right, bottom-left
        # Reorder to match ArUco convention
        corners_reordered = np.array(
            [
                result.corners[3],  # top-left
                result.corners[2],  # top-right
                result.corners[1],  # bottom-right
                result.corners[0],  # bottom-left
            ]
        ).astype(np.float32)

        corners.append(corners_reordered)
        marker_ids.append(result.tag_id)

    return corners, marker_ids, image_size


def _detect_apriltag_markers_in_project(
    project,
    tag_family: str,
    progress_bar: bool,
) -> Dict:
    """
    Detect AprilTag markers in all project images using pupil-apriltags.
    Fixed version that creates detector once and reuses it.
    """
    # Load images
    image_ids = list(project.images.keys())
    images = []

    for image_id in tqdm(
        image_ids,
        desc=f"Loading images for AprilTag {tag_family}",
        disable=not progress_bar,
    ):
        image = project.load_image_by_id(image_id)
        images.append(image)

    # Create the detector ONCE outside the loop
    detector = None
    try:
        detector = AprilTagDetector(families=tag_family)
        logging.info(f"Created AprilTag detector for family: {tag_family}")

        # Sequential AprilTag detection with reused detector
        results = []
        for image in tqdm(
            images,
            desc=f"Detecting AprilTag ({tag_family})",
            disable=not progress_bar,
        ):
            corners, marker_ids, image_size = detect_apriltag_markers_in_image(
                image, detector
            )
            results.append((corners, marker_ids, image_size))

    except Exception as e:
        logging.error(f"AprilTag detection failed: {e}")
        results = [(None, None, image.shape) for image in images]
    finally:
        # Explicitly clean up detector
        if detector is not None:
            try:
                del detector
            except:
                pass  # Ignore cleanup errors

    return _process_detection_results(project, image_ids, results, MarkerType.APRILTAG)


def _detect_apriltag_markers_in_project(
    project,
    tag_family: str,
    progress_bar: bool,
) -> Dict:
    """
    Detect AprilTag markers in all project images using pupil-apriltags.
    Fixed version that creates detector once and reuses it.
    """
    # Load images
    image_ids = list(project.images.keys())
    images = []

    for image_id in tqdm(
        image_ids,
        desc=f"Loading images for AprilTag {tag_family}",
        disable=not progress_bar,
    ):
        image = project.load_image_by_id(image_id)
        images.append(image)

    # Create the detector ONCE outside the loop
    detector = None
    try:
        detector = AprilTagDetector(families=tag_family)
        logging.info(f"Created AprilTag detector for family: {tag_family}")

        # Sequential AprilTag detection with reused detector
        results = []
        for image in tqdm(
            images,
            desc=f"Detecting AprilTag ({tag_family})",
            disable=not progress_bar,
        ):
            corners, marker_ids, image_size = detect_apriltag_markers_in_image(
                image, detector
            )
            results.append((corners, marker_ids, image_size))

    except Exception as e:
        logging.error(f"AprilTag detection failed: {e}")
        results = [(None, None, image.shape) for image in images]
    finally:
        # Explicitly clean up detector
        if detector is not None:
            try:
                del detector
            except:
                pass  # Ignore cleanup errors

    return _process_detection_results(project, image_ids, results, MarkerType.APRILTAG)


def _detect_aruco_sequential(images, detector, dict_type, progress_bar):
    """Sequential ArUco marker detection (fallback when multiprocessing fails)."""
    results = []
    for image in tqdm(
        images,
        desc=f"Detecting ArUco (dict {dict_type}) - sequential",
        disable=not progress_bar,
    ):
        image_size = image.shape
        corners, marker_ids, _ = detector.detectMarkers(image)
        if marker_ids is None:
            results.append((None, None, image_size))
        else:
            results.append((corners, marker_ids, image_size))
    return results


def _process_detection_results(project, image_ids, results, marker_type):
    """Process detection results into unified format."""
    detection_data = {}
    detected_ids = []

    for image_idx, image_id in enumerate(image_ids):
        image = project.images[image_id]
        camera = project.cameras[image.camera_id]
        result = results[image_idx]

        # Calculate scaling ratios
        ratio_x = camera.width / result[2][1]
        ratio_y = camera.height / result[2][0]

        detection_data[image_id] = {
            "marker_corners": [],
            "marker_ids": [],
            "corner_pixels": [],
        }

        if result[0] is not None and result[1] is not None:
            for corner_set, marker_id in zip(result[0], result[1]):
                if marker_type == MarkerType.ARUCO:
                    # ArUco format: tuple with nested arrays
                    pixels = corner_set[0]
                    # Scale corners to camera resolution
                    scaled_corners = (
                        np.expand_dims(
                            np.vstack(
                                [
                                    corner_set[0, :, 0] * ratio_y,
                                    corner_set[0, :, 1] * ratio_x,
                                ]
                            ).T,
                            axis=0,
                        ),
                    )
                    marker_id_val = marker_id[0]

                elif marker_type == MarkerType.APRILTAG:
                    # AprilTag format: direct numpy array
                    pixels = corner_set
                    # Scale corners to camera resolution
                    scaled_corners = np.vstack(
                        [
                            corner_set[:, 0] * ratio_y,
                            corner_set[:, 1] * ratio_x,
                        ]
                    ).T
                    marker_id_val = marker_id

                detection_data[image_id]["marker_corners"].append(scaled_corners)
                detection_data[image_id]["marker_ids"].append(marker_id_val)
                detection_data[image_id]["corner_pixels"].append(pixels)
                detected_ids.append(marker_id_val)

    unique_ids = list(set(detected_ids))
    if unique_ids:
        logging.info(f"{marker_type.value}: Detected marker IDs: {sorted(unique_ids)}")
    else:
        logging.warning(f"{marker_type.value}: No markers detected")

    return detection_data


def _calculate_3d_positions_robust(
    project,
    marker_type: MarkerType,
    detection_data: Dict,
    min_detections: int,
    ransac_config: Dict,
) -> Dict:
    """
    Calculate 3D positions for detected markers with minimum detection threshold and RANSAC.
    """
    # Collect all unique marker IDs
    all_ids = set()
    for image_data in detection_data.values():
        all_ids.update(image_data["marker_ids"])

    # Count detections per marker ID
    detection_counts = {marker_id: 0 for marker_id in all_ids}
    for image_data in detection_data.values():
        for marker_id in image_data["marker_ids"]:
            detection_counts[marker_id] += 1

    # Filter markers that don't meet minimum detection threshold
    valid_ids = [
        marker_id
        for marker_id, count in detection_counts.items()
        if count >= min_detections
    ]

    logging.info(f"{marker_type.value}: {len(all_ids)} total markers detected")
    logging.info(
        f"{marker_type.value}: {len(valid_ids)} markers meet minimum detection threshold ({min_detections})"
    )

    if not valid_ids:
        logging.warning(
            f"{marker_type.value}: No markers meet minimum detection threshold"
        )
        return {}

    # Collect ray casting data for valid markers only
    ray_data = {
        marker_id: {"P0": [], "N": [], "image_ids": [], "corner_pixels": []}
        for marker_id in valid_ids
    }

    # Process each image
    for image_id, image_data in detection_data.items():
        if not image_data["marker_corners"]:
            continue

        image = project.images[image_id]
        camera = project.cameras[image.camera_id]

        # Process each detected marker in this image
        for corners, marker_id, pixels in zip(
            image_data["marker_corners"],
            image_data["marker_ids"],
            image_data["corner_pixels"],
        ):
            # Only process valid markers
            if marker_id not in valid_ids:
                continue

            # Cast rays for this marker
            p0, n = ray_cast_marker_corners(
                extrinsics=image.world_extrinsics,
                intrinsics=camera.intrinsics.K,
                corners=corners,
            )

            ray_data[marker_id]["P0"].append(p0)
            ray_data[marker_id]["N"].append(n)
            ray_data[marker_id]["image_ids"].append(image_id)
            ray_data[marker_id]["corner_pixels"].append(pixels)

    # Calculate 3D positions using RANSAC
    marker_results = {}

    for marker_id, data in ray_data.items():
        if len(data["P0"]) == 0:
            continue

        # Convert to numpy arrays
        P0_array = np.array(data["P0"])  # (n_detections, 3)
        N_array = np.array(data["N"])  # (n_detections, 4, 3)

        # Process each corner separately with RANSAC
        corners_3d = []
        corner_inlier_counts = []

        for corner_idx in range(4):
            # Extract rays for this corner from all detections
            corner_centers = P0_array  # All camera centers
            corner_directions = N_array[
                :, corner_idx, :
            ]  # Ray directions for this corner

            # Use RANSAC to find robust 3D position
            corner_3d, inliers = ransac_ray_intersection(
                corner_centers,
                corner_directions,
                max_iterations=ransac_config["max_iterations"],
                distance_threshold=ransac_config["distance_threshold"],
                min_inliers=ransac_config["min_inliers"],
            )

            if corner_3d is not None:
                corners_3d.append(corner_3d)
                corner_inlier_counts.append(len(inliers))
                logging.debug(
                    f"{marker_type.value}, ID {marker_id}, Corner {corner_idx}: "
                    f"{len(inliers)}/{len(corner_centers)} inliers"
                )
            else:
                # RANSAC failed - skip this corner entirely
                logging.warning(
                    f"{marker_type.value}, ID {marker_id}, Corner {corner_idx}: "
                    f"RANSAC failed - insufficient inliers or poor geometry"
                )
                continue

        # Only proceed if we have all 4 corners
        if len(corners_3d) == 4:
            corners_3d = np.array(corners_3d)
            center_xyz = np.mean(corners_3d, axis=0)

            marker_results[marker_id] = {
                "corners_3d": corners_3d,
                "center_xyz": center_xyz,
                "image_ids": data["image_ids"],
                "corner_pixels": data["corner_pixels"],
                "marker_type": marker_type.value,
                "total_detections": len(P0_array),
                "corner_inlier_counts": corner_inlier_counts,
                "detection_quality": np.mean(corner_inlier_counts) / len(P0_array),
            }

            logging.info(
                f"{marker_type.value}, ID {marker_id}: "
                f"detections={len(P0_array)}, "
                f"avg_inliers={np.mean(corner_inlier_counts):.1f}, "
                f"quality={marker_results[marker_id]['detection_quality']:.2f}"
            )
        else:
            logging.warning(
                f"{marker_type.value}, ID {marker_id}: "
                f"Only {len(corners_3d)}/4 corners successfully localized"
            )

    return marker_results
