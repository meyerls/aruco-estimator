import json
import logging
import os
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from aruco_estimator.aruco_localizer import ArucoLocalizer
from aruco_estimator.utils import get_normalization_transform
from aruco_estimator.visualization import VisualizationModel


def register(
    project: str,
    aruco_size: float = 0.2,
    dict_type: int = cv2.aruco.DICT_4X4_50,
    show_original: bool = False,
    show: bool = False,
    target_id: int = 0,
    export_tags: bool = False,
    export_path: str = None,
):
    """
    Normalize COLMAP poses relative to ArUco marker.

    Args:
        project: Path to COLMAP project
        aruco_size: Size of ArUco marker in meters
        dict_type: ArUco dictionary type
        show_original: Whether to show original poses in visualization
        show: Whether to show the result
        target_id: ID of ArUco marker to use as origin (default: 0)
        export_tags: Whether to export tag positions (default: False)
        export_path: Path to export tag positions (default: project_path/aruco_tags.json)
    """
    from aruco_estimator.sfm.colmap import COLMAPProject
    project_path = Path(project)
    logging.basicConfig(level=logging.INFO)

    # Set default export path if not provided
    if export_tags and export_path is None:
        export_path = os.path.join(project_path, "aruco_tags.json")

    # Load COLMAP project using new interface
    logging.info("Loading COLMAP project...")
    project = COLMAPProject(str(project_path))
    
    # Store original project state if needed for visualization
    original_project = None
    if show_original:
        original_project = deepcopy(project)

    # Run ArUco detection
    logging.info("Detecting ArUco markers...")
    aruco_localizer = ArucoLocalizer(
        project=project,
        dict_type=dict_type,
        target_id=target_id,
    )
    aruco_distance, aruco_corners_3d = aruco_localizer.run()
    logging.info(f"Target ArUco ID: {target_id}")
    logging.info(f"ArUco 3d points: {aruco_corners_3d}")
    logging.info(f"ArUco marker distance: {aruco_distance}")

    # Calculate 3D positions for all detected ArUco markers
    all_aruco_positions = None
    original_aruco_positions = None
    if export_tags:
        logging.info("Calculating positions for all detected ArUco markers...")
        all_aruco_positions = aruco_localizer.get_all_aruco_positions()
        if show_original:
            original_aruco_positions = deepcopy(all_aruco_positions)
        logging.info(f"Found {len(all_aruco_positions)} ArUco markers")

    # Calculate normalization transform with scaling
    transform = get_normalization_transform(aruco_corners_3d, aruco_size)
    # transform = np.eye(4)

    # Apply normalization to the project
    logging.info("Normalizing poses and 3D points...")
    project.transform(transform)

    # Verify the scaling worked correctly by measuring the marker in the transformed space
    transformed_corners = np.array(
        [
            (transform @ np.append(corner, 1))[:3]
            / (transform @ np.append(corner, 1))[3]
            for corner in aruco_corners_3d
        ]
    )

    # Measure transformed marker dimensions
    transformed_width = np.linalg.norm(transformed_corners[0] - transformed_corners[1])
    transformed_height = np.linalg.norm(transformed_corners[0] - transformed_corners[3])
    logging.info(
        f"Normalized ArUco dimensions: width={transformed_width:.4f}, height={transformed_height:.4f}"
    )
    logging.info(f"Target ArUco size: {aruco_size}")

    if show:
        # Create visualization model
        model = VisualizationModel()
        model.create_window()

        # Add original data in gray if requested
        if show_original and original_project:
            model.add_project(
                original_project,
                points_config={"color": [0.7, 0.7, 0.7]},
                cameras_config={"scale": 0.25, 
                                "color": [0.7, 0.7, 0.7],
                                "show_images": False}
            )

        # Add transformed data
        model.add_project(
            project,
            points_config={},  # Use default colors from point cloud
            cameras_config={"scale": 0.25, "color": [1, 0, 0]}  # Red cameras
        )

        # Add coordinate frames
        if show_original:
            model.add_coordinate_frame(size=1.0, transform=transform)  # Transformed frame
        model.add_coordinate_frame(size=2.0)  # True coordinate frame

        # Add ArUco markers
        if show_original:
            model.add_aruco_marker(aruco_corners_3d, color=[1, 0, 1])  # Magenta for original
        model.add_aruco_marker(transformed_corners, color=[0, 1, 1])  # Cyan for transformed

        # Add all detected ArUco markers
        if export_tags and all_aruco_positions:
            logging.info("Visualizing all detected ArUco markers...")

            # Filter out target marker and prepare data
            other_markers_original = {
                mid: corners for mid, corners in original_aruco_positions.items() 
                if mid != target_id
            } if show_original and original_aruco_positions else {}

            # Transform other markers to new coordinate system
            other_markers_transformed = {}
            for aruco_id, corners in all_aruco_positions.items():
                if aruco_id == target_id:
                    continue
                    
                # Transform marker corners to new coordinate system
                transformed_marker_corners = np.array([
                    (transform @ np.append(corner, 1))[:3]
                    / (transform @ np.append(corner, 1))[3]
                    for corner in corners
                ])
                other_markers_transformed[aruco_id] = transformed_marker_corners

            # Add original markers
            if other_markers_original:
                model.add_aruco_markers(other_markers_original)

            # Add transformed markers
            if other_markers_transformed:
                model.add_aruco_markers(other_markers_transformed)

        # Show visualization
        model.show()

    # Export tag positions if requested
    if export_tags and all_aruco_positions:
        logging.info("Exporting ArUco tag positions...")
        # Transform ArUco corners to new coordinate system
        transformed_aruco_positions = {}
        for aruco_id, corners in all_aruco_positions.items():
            # Transform each corner using homogeneous coordinates
            transformed_corners = np.array(
                [
                    (transform @ np.append(corner, 1))[:3]
                    / (transform @ np.append(corner, 1))[3]
                    for corner in corners
                ]
            )

            # Verify the scaling for each marker
            if aruco_id == target_id:
                # This is our reference marker, should match aruco_size
                width = np.linalg.norm(transformed_corners[0] - transformed_corners[1])
                height = np.linalg.norm(transformed_corners[0] - transformed_corners[3])
                logging.info(
                    f"Exported target ArUco marker {aruco_id}: width={width:.4f}, height={height:.4f}"
                )

            transformed_aruco_positions[int(aruco_id)] = transformed_corners.tolist()

        # Save to JSON file
        with open(export_path, "w") as f:
            json.dump(
                {
                    "aruco_tags": transformed_aruco_positions,
                    "aruco_size": aruco_size,
                    "target_id": target_id,
                },
                f,
                indent=2,
            )
        logging.info(f"ArUco tag positions exported to {export_path}")

    # Save transformed data
    logging.info("Saving normalized data...")
    output_dir = project_path / "normalized" / "sparse"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save transformed data using the project's save method
    project.save(str(output_dir))

    logging.info("Done! Normalized data saved to normalized/sparse/")