import logging
from copy import deepcopy
from pathlib import Path
import cv2
from aruco_estimator.utils import get_transformation_between_clouds, get_corners_at_origin
from aruco_estimator.visualization import VisualizationModel

import json
import os
import numpy as np

def register(
    project,
    aruco_size: float = 0.2,
    dict_type: int = cv2.aruco.DICT_4X4_50, 
    show_original: bool = False,
    show: bool = False,
    target_id: int = 0,
    export_path: str = None,
):
    """
    Normalize COLMAP poses relative to ArUco marker.
    
    Args:
        project: SfM project instance (not path)
        aruco_size: Size of ArUco marker in meters
        dict_type: ArUco dictionary size 
        show_original: Whether to show original poses in visualization
        show: Whether to show the result
        target_id: ID of ArUco marker to use as origin (default: 0)
        export_tags: Whether to export tag positions (default: False)
        export_path: Path to export tag positions (default: project_path/aruco_tags.json)
    """
    export_tags = True
    # Run ArUco detection with proper dictionary type
    logging.info(f"Detecting ArUco markers using {dict_type}x{dict_type} dictionary...")
    
    # Pass the dict_type parameter to detect_markers
    aruco_results = project.detect_markers(dict_type=dict_type)
    
    if not aruco_results:
        logging.warning("No ArUco markers detected!")
        return
    # Check if target marker was found
    if target_id not in aruco_results:
        available_ids = list(aruco_results.keys())
        logging.warning(f"Target marker ID {target_id} not found. Available IDs: {available_ids}")
        return
    
    # Get 3D corners for normalization
    target_corners_3d = aruco_results[target_id]
    # logging.info(f"Using marker {target_id} for normalization (distance: {target_distance:.3f})")
    print(target_corners_3d) 
    # Calculate normalization transform with scaling
    transform = get_transformation_between_clouds(target_corners_3d, get_corners_at_origin(side_length=aruco_size))
    
    # Store original project state if needed for visualization
    original_project = None
    if show_original:
        original_project = deepcopy(project)
    
    # Apply normalization to the project
    logging.info("Normalizing poses and 3D points...")
    project.transform(transform)
    
    if show:
        model = VisualizationModel()
        model.create_window()
        
        # Add original data in gray if requested (show first so it's in background)
        if show_original:
            model.add_project(
                original_project,
                points_config={"color": [0.7, 0.7, 0.7]},
                cameras_config={
                    "scale": 0.25, 
                    "color": [0.7, 0.7, 0.7],
                    "show_images": False
                },
                markers_config={
                    "show_detection_lines": False,
                    "corner_size": 0.03,
                }
            )
        
        # Add transformed data (foreground)
        model.add_project(
            project,
            points_config={},  # Use default colors from point cloud
            cameras_config={"scale": 0.25, "color": [1, 0, 0]},  # Red cameras
            markers_config={
                "show_detection_lines": True,
                "detection_line_color": [0, 1, 0],  # Green detection lines
                "corner_size": 0.05,
            }
        )
        
        # Add coordinate frame at origin
        model.add_coordinate_frame(size=aruco_size)
        
        # Show visualization
        model.show()
    
    # Export tag positions if requested
    if export_tags:
        
        logging.info("Exporting ArUco tag positions...")
      
        # Get all marker positions after transformation (they're already transformed in the project)
        all_markers = {}
        for dict_type_key, markers_dict in project.markers.items():
            if dict_type_key == dict_type:  # Only export markers from the dict we used
                for marker_id, marker in markers_dict.items():
                    try:
                        # Get transformed 3D corners from the project (already transformed)
                        transformed_corners = project.get(dict_type_key, {}).get(marker_id, None)
                        
                        all_markers[int(marker_id)] = {
                            "corners_3d": transformed_corners.tolist(),
                            "center_3d": transformed_corners.mean(axis=0).tolist(),
                        }
                    except Exception as e:
                        logging.warning(f"Could not export marker {marker_id}: {e}")
        
        export_data = {
            "aruco_tags": all_markers,
            "aruco_size": aruco_size,
            "target_id": target_id,
            "dict_type": f"{dict_type}x{dict_type}",
            "normalization_transform": transform.tolist()
        }
        if export_path is None:
            # Use project directory if available, otherwise current directory
            if hasattr(project, 'project_path'):
                base_path = project.project_path
            else:
                base_path = Path.cwd()
            export_path = base_path / "aruco_tags.json"
        
        # Save to JSON file
        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)
        
        logging.info(f"ArUco tag positions exported to {export_path}")
    
    # Save transformed data if project has save method
    logging.info("Saving normalized data...")
    output_dir = Path("normalized") / "sparse"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save transformed data using the project's save method
    project.save(str(output_dir))
    logging.info(f"Normalized data saved to {output_dir}")

    logging.info("Registration complete!")
    # return transform, aruco_results