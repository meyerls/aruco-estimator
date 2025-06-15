import logging
from copy import deepcopy
from pathlib import Path

import cv2
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

    # Run ArUco detection
    logging.info("Detecting ArUco markers...")
    
    aruco_corners_3d = project.detect_markers()
    # aruco_corners_3d = aruco_corners_3d
    logging.info(f"ArUco 3d points: {aruco_corners_3d}")

    # Calculate normalization transform with scaling
    # transform = get_normalization_transform(aruco_corners_3d, aruco_size)
    # transform = np.eye(4)
    
    # Store original project state if needed for visualization
    original_project = None
    if show_original:
        original_project = deepcopy(project)
    # Apply normalization to the project
    logging.info("Normalizing poses and 3D points...")
    # project.transform(transform)

    if show:
        model = VisualizationModel()
        model.create_window()

        # Add transformed data
        model.add_project(
            project,
            points_config={},  # Use default colors from point cloud
            cameras_config={"scale": 0.25, "color": [1, 0, 0]}  # Red cameras
        )

        # Show visualization
        model.show()
        
        # Add original data in gray if requested
        if show_original:
            model.add_project(
                original_project,
                points_config={"color": [0.7, 0.7, 0.7]},
                cameras_config={"scale": 0.25, 
                                "color": [0.7, 0.7, 0.7],
                                "show_images": False},
                # origin_size=1
            )


    # # Export tag positions if requested
    # if export_tags:
    #     logging.info("Exporting ArUco tag positions...")
    #     if export_tags and export_path is None:
    #         export_path = os.path.join(project_path, "aruco_tags.json")

    #     # Save to JSON file
    #     with open(export_path, "w") as f:
    #         json.dump(
    #             {
    #                 # "aruco_tags": transformed_aruco_positions,
    #                 "aruco_size": aruco_size,
    #                 "target_id": target_id,
    #             },
    #             f,
    #             indent=2,
    #         )
    #     logging.info(f"ArUco tag positions exported to {export_path}")

    # # Save transformed data
    # logging.info("Saving normalized data...")
    # output_dir = project_path / "normalized" / "sparse"
    # output_dir.mkdir(parents=True, exist_ok=True)

    # # Save transformed data using the project's save method
    # project.save(str(output_dir))

    # logging.info("Done! Normalized data saved to normalized/sparse/")