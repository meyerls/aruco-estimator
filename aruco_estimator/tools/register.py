import logging
import cv2
from aruco_estimator.utils import (
    get_transformation_between_clouds,
    get_corners_at_origin,
)


def register(
    project,
    marker_size: float = 0.2,
    marker_config: dict = None,
    target_id: int = 0,
    # Legacy parameters for backward compatibility
    aruco_size: float = None,
    dict_type: int = None,
):
    """
    Normalize COLMAP poses relative to ArUco marker or AprilTag.

    This function performs the core registration logic by:
    1. Detecting markers (ArUco or AprilTag) in the project
    2. Finding the target marker
    3. Computing the transformation to align the target marker with the origin
    4. Applying the transformation to the project

    Args:
        project: SfM project instance (not path)
        marker_size: Size of marker in meters
        marker_config: Dictionary with marker configuration:
                      - For ArUco: {'type': 'aruco', 'dict_type': cv2.aruco.DICT_4X4_50}
                      - For AprilTag: {'type': 'apriltag', 'family': 'tag36h11'}
        target_id: ID of marker to use as origin (default: 0)

        # Legacy parameters (for backward compatibility):
        aruco_size: (deprecated) Use marker_size instead
        dict_type: (deprecated) Use marker_config instead

    Returns:
        tuple: (transformed_project, transformation_matrix, marker_results) or (None, None, None) if failed
    """
    # Handle legacy parameters for backward compatibility
    if aruco_size is not None:
        logging.warning(
            "Parameter 'aruco_size' is deprecated. Use 'marker_size' instead."
        )
        marker_size = aruco_size

    if dict_type is not None and marker_config is None:
        logging.warning(
            "Parameter 'dict_type' is deprecated. Use 'marker_config' instead."
        )
        marker_config = {"type": "aruco", "dict_type": dict_type}

    # Default to ArUco if no marker_config provided
    if marker_config is None:
        marker_config = {"type": "aruco", "dict_type": cv2.aruco.DICT_4X4_50}
        logging.info("No marker_config provided, defaulting to ArUco DICT_4X4_50")

    # Validate marker_config
    if not isinstance(marker_config, dict) or "type" not in marker_config:
        logging.error("marker_config must be a dictionary with 'type' key")
        return None, None, None

    marker_type = marker_config["type"].lower()

    if marker_type == "aruco":
        if "dict_type" not in marker_config:
            logging.error("ArUco marker_config must include 'dict_type'")
            return None, None, None
        marker_results = _detect_aruco_markers(project, marker_config["dict_type"])

    elif marker_type == "apriltag":
        if "family" not in marker_config:
            logging.error("AprilTag marker_config must include 'family'")
            return None, None, None
        marker_results = _detect_apriltag_markers(project, marker_config["family"])

    else:
        logging.error(
            f"Unsupported marker type: {marker_type}. Supported types: 'aruco', 'apriltag'"
        )
        return None, None, None

    if not marker_results:
        logging.warning(f"No {marker_type} markers detected!")
        return None, None, None

    # Check if target marker was found
    if target_id not in marker_results:
        available_ids = list(marker_results.keys())
        logging.warning(
            f"Target marker ID {target_id} not found. Available IDs: {available_ids}"
        )
        return None, None, None

    # Get 3D corners for normalization
    if marker_type == "aruco":
        target_corners_3d = marker_results[target_id]
    elif marker_type == "apriltag":
        # AprilTag results have different structure
        target_corners_3d = marker_results[target_id]["corners_3d"]

    logging.info(f"Using {marker_type} marker {target_id} for normalization")
    logging.debug(f"Target corners 3D: {target_corners_3d}")

    # Calculate normalization transform with scaling
    transform = get_transformation_between_clouds(
        target_corners_3d, get_corners_at_origin(side_length=marker_size)
    )

    # Apply normalization to the project
    logging.info("Normalizing poses and 3D points...")
    project.transform(transform)

    logging.info("Registration complete!")
    return project, transform, marker_results


def _detect_aruco_markers(project, dict_type):
    """
    Detect ArUco markers in the project.

    Args:
        project: SfM project instance
        dict_type: ArUco dictionary type

    Returns:
        Dictionary mapping marker_id -> 3d_corners
    """
    logging.info(f"Detecting ArUco markers using dictionary type {dict_type}...")

    # Pass the dict_type parameter to detect_markers
    aruco_results = project.detect_markers(dict_type=dict_type)

    return aruco_results


def _detect_apriltag_markers(project, tag_family):
    """
    Detect AprilTag markers in the project.

    Args:
        project: SfM project instance
        tag_family: AprilTag family (e.g., 'tag36h11')

    Returns:
        Dictionary mapping marker_id -> marker_data
    """
    logging.info(f"Detecting AprilTag markers using family {tag_family}...")

    # Run AprilTag detection using the project's method
    apriltag_results = project.detect_apriltag_markers(
        tag_family=tag_family,
        progress_bar=True,
        min_detections=3,
    )

    return apriltag_results
