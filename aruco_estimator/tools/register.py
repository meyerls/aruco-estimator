import logging
import cv2
import numpy as np
from aruco_estimator.utils import (
    get_transformation_between_clouds,
    get_corners_at_origin,
)


def register(
    project,
    aruco_size: float = 0.2,
    dict_type: int = cv2.aruco.DICT_4X4_50,
    target_id: int = 0,
):
    """
    Normalize COLMAP poses relative to ArUco marker.

    This function performs the core registration logic by:
    1. Detecting ArUco markers in the project
    2. Finding the target marker
    3. Computing the transformation to align the target marker with the origin
    4. Applying the transformation to the project

    Args:
        project: SfM project instance (not path)
        aruco_size: Size of ArUco marker in meters
        dict_type: ArUco dictionary size
        target_id: ID of ArUco marker to use as origin (default: 0)

    Returns:
        tuple: (transformed_project, transformation_matrix, aruco_results) or (None, None, None) if failed
    """
    # Run ArUco detection with proper dictionary type
    logging.info(f"Detecting ArUco markers using dictionary type {dict_type}...")

    # Pass the dict_type parameter to detect_markers
    aruco_results = project.detect_markers(dict_type=dict_type)

    if not aruco_results:
        logging.warning("No ArUco markers detected!")
        return None, None, None

    # Check if target marker was found
    if target_id not in aruco_results:
        available_ids = list(aruco_results.keys())
        logging.warning(
            f"Target marker ID {target_id} not found. Available IDs: {available_ids}"
        )
        return None, None, None

    # Get 3D corners for normalization
    target_corners_3d = aruco_results[target_id]
    logging.info(f"Using marker {target_id} for normalization")
    logging.debug(f"Target corners 3D: {target_corners_3d}")

    # Calculate normalization transform with scaling
    transform = get_transformation_between_clouds(
        target_corners_3d, get_corners_at_origin(side_length=aruco_size)
    )

    # Apply normalization to the project
    logging.info("Normalizing poses and 3D points...")
    project.transform(transform)

    logging.info("Registration complete!")
    return project, transform, aruco_results
