
from aruco_estimator.tools.register import register

import logging
from copy import deepcopy
from pathlib import Path
import cv2
from aruco_estimator.utils import get_transformation_between_clouds, get_corners_at_origin
from aruco_estimator.visualization import VisualizationModel
from aruco_estimator.sfm.common import SfmProjectBase

import json
import os
import numpy as np

import numpy as np
import logging
import numpy as np
import logging
from typing import Dict, List, Tuple

def merge_projects(projects: List['SfmProjectBase'], id_strategy: str = "offset") -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Merge multiple SfM projects by reassigning all IDs.
    
    :param projects: List of SfM projects to merge
    :param id_strategy: Strategy for handling ID conflicts ("offset", "prefix")
    :return: Tuple of (_cameras, _images, _points3D, _markers) dictionaries
    """
    if len(projects) < 2:
        raise ValueError("Need at least 2 projects to merge")
    
    merged_cameras = {}
    merged_images = {}
    merged_points3D = {}
    merged_markers = {}
    
    # Calculate ID assignments for all projects
    id_assignments = _calculate_all_id_assignments(projects, id_strategy)
    
    # Merge each project
    for idx, project in enumerate(projects):
        assignment = id_assignments[idx]
        
        # Merge cameras
        for old_id, new_id in assignment['cameras'].items():
            camera = project.cameras[old_id]
            merged_cameras[new_id] = camera._replace(id=new_id)
        
        # Merge images
        for old_id, new_id in assignment['images'].items():
            image = project.images[old_id]
            new_camera_id = assignment['cameras'][image.camera_id]
            
            # Update point3D_ids references
            new_point3D_ids = None
            if image.point3D_ids is not None:
                new_point3D_ids = []
                for old_pt_id in image.point3D_ids:
                    if old_pt_id == -1:
                        new_point3D_ids.append(-1)
                    elif old_pt_id in assignment['points']:
                        new_point3D_ids.append(assignment['points'][old_pt_id])
                    else:
                        new_point3D_ids.append(-1)
                new_point3D_ids = np.array(new_point3D_ids)
            
            merged_images[new_id] = image._replace(
                id=new_id,
                camera_id=new_camera_id,
                point3D_ids=new_point3D_ids
            )
        
        # Merge 3D points
        for old_id, new_id in assignment['points'].items():
            point = project.points3D[old_id]
            merged_points3D[new_id] = point._replace(id=new_id)
        
        # Merge markers
        if hasattr(project, '_markers') and project._markers:
            for dict_type, markers in project.markers.items():
                if dict_type not in merged_markers:
                    merged_markers[dict_type] = {}
                
                for marker_id, marker in markers.items():
                    if marker_id in merged_markers[dict_type]:
                        # Combine observations for same ArUco marker
                        existing = merged_markers[dict_type][marker_id]
                        combined_image_ids = list(existing.image_ids) + list(marker.image_ids)
                        combined_point2D_idxs = list(existing.point2D_idxs) + list(marker.point2D_idxs)
                        avg_xyz = (existing.xyz + marker.xyz) / 2
                        avg_corners = (existing.corners_3d + marker.corners_3d) / 2
                        
                        merged_markers[dict_type][marker_id] = existing._replace(
                            xyz=avg_xyz,
                            corners_3d=avg_corners,
                            image_ids=combined_image_ids,
                            point2D_idxs=combined_point2D_idxs
                        )
                    else:
                        merged_markers[dict_type][marker_id] = marker
    
    logging.info(f"Merged {len(projects)} projects: {len(merged_cameras)} cameras, "
                f"{len(merged_images)} images, {len(merged_points3D)} points")
    
    return merged_cameras, merged_images, merged_points3D, merged_markers

def _calculate_all_id_assignments(projects: List['SfmProjectBase'], id_strategy: str) -> List[Dict]:
    """Calculate new ID assignments for all projects to avoid conflicts."""
    
    assignments = []
    
    if id_strategy == "offset":
        # Sequential assignment
        next_camera_id = 1
        next_image_id = 1
        next_point_id = 1
        
        for project in projects:
            assignment = {'cameras': {}, 'images': {}, 'points': {}}
            
            for old_id in sorted(project.cameras.keys()):
                assignment['cameras'][old_id] = next_camera_id
                next_camera_id += 1
            
            for old_id in sorted(project.images.keys()):
                assignment['images'][old_id] = next_image_id
                next_image_id += 1
            
            for old_id in sorted(project.points3D.keys()):
                assignment['points'][old_id] = next_point_id
                next_point_id += 1
            
            assignments.append(assignment)
    
    elif id_strategy == "prefix":
        # Use project index as prefix
        for proj_idx, project in enumerate(projects):
            max_id = max(
                max(project.cameras.keys()) if project.cameras else 0,
                max(project.images.keys()) if project.images else 0,
                max(project.points3D.keys()) if project.points3D else 0
            )
            multiplier = 10 ** (len(str(max_id)) + 2)
            base_offset = proj_idx * multiplier
            
            assignment = {
                'cameras': {old_id: base_offset + old_id for old_id in project.cameras.keys()},
                'images': {old_id: base_offset + old_id for old_id in project.images.keys()},
                'points': {old_id: base_offset + old_id for old_id in project.points3D.keys()}
            }
            assignments.append(assignment)
    
    else:
        raise ValueError(f"Unknown ID strategy: {id_strategy}")
    
    return assignments

def align_projects(projects: List['SfmProjectBase'], 
                  target_id: int = 0,
                  min_markers: int = 3,
                  show: bool = False) -> List['SfmProjectBase']:
    """
    Align multiple projects using common ArUco markers.
    
    :param projects: List of SfM projects to align
    :param target_id: Index of target project (root coordinate system)
    :param min_markers: Minimum number of markers needed for alignment
    :param show: Whether to show visualization
    :return: List of aligned projects
    """
    if target_id >= len(projects):
        raise ValueError(f"target_id {target_id} out of range for {len(projects)} projects")
    
    # Find marker links between all project pairs
    links = {}
    for i in range(len(projects)):
        for j in range(len(projects)):
            if i == j:
                continue
            
            common_markers = _find_common_markers(projects[i], projects[j])
            if len(common_markers) >= min_markers:
                links[(i, j)] = common_markers
                logging.info(f"Found {len(common_markers)} common markers between projects {i} and {j}")
    
    if not links:
        raise ValueError("No projects have sufficient common markers for alignment")
    
    # Check connectivity - ensure all projects can be reached from target
    connected_projects = _find_connected_projects(links, target_id, len(projects))
    disconnected = set(range(len(projects))) - connected_projects
    if disconnected:
        logging.warning(f"Projects {disconnected} cannot be connected to target project {target_id}")
    
    # Calculate transformations for each project relative to target
    transformations = _calculate_transformations(projects, links, target_id)
    
    # Apply transformations
    aligned_projects = []
    for i, project in enumerate(projects):
        if i == target_id:
            # Target project stays in original coordinate system
            aligned_projects.append(project)
        else:
            # Apply transformation
            aligned_project = deepcopy(project)
            if i in transformations:
                aligned_project.transform(transformations[i])
                logging.info(f"Applied transformation to project {i}")
            else:
                logging.warning(f"No transformation found for project {i}")
                aligned_project = project
            aligned_projects.append(aligned_project)
    
    return aligned_projects

def _find_common_markers(proj_A: 'SfmProjectBase', proj_B: 'SfmProjectBase') -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Find common ArUco markers between two projects."""
    common_markers = {}
    
    if not (hasattr(proj_A, '_markers') and hasattr(proj_B, '_markers')):
        return common_markers
    
    for dict_type in proj_A.markers:
        if dict_type not in proj_B.markers:
            continue
        
        for marker_id in proj_A.markers[dict_type]:
            if marker_id in proj_B.markers[dict_type]:
                marker_A = proj_A.markers[dict_type][marker_id]
                marker_B = proj_B.markers[dict_type][marker_id]
                # Store corners from both projects for this marker
                common_markers[marker_id] = (marker_A.corners_3d, marker_B.corners_3d)
    
    return common_markers

def _find_connected_projects(links: Dict[Tuple[int, int], Dict], target_id: int, num_projects: int) -> set:
    """Find all projects connected to target through marker links."""
    connected = {target_id}
    changed = True
    
    while changed:
        changed = False
        for (i, j), _ in links.items():
            if i in connected and j not in connected:
                connected.add(j)
                changed = True
            elif j in connected and i not in connected:
                connected.add(i)
                changed = True
    
    return connected

def _calculate_transformations(projects: List['SfmProjectBase'], 
                             links: Dict[Tuple[int, int], Dict],
                             target_id: int) -> Dict[int, np.ndarray]:
    """Calculate transformation matrices to align projects to target."""
    transformations = {target_id: np.eye(4)}  # Target uses identity
    calculated = {target_id}
    
    # Use breadth-first search to calculate transformations
    to_process = [target_id]
    
    while to_process:
        current = to_process.pop(0)
        
        # Find all projects connected to current
        for (i, j), common_markers in links.items():
            if i == current and j not in calculated:
                # Calculate transformation from j to i (current)
                transform = _compute_marker_transformation(common_markers, from_proj=j, to_proj=i)
                # Chain with current's transformation
                transformations[j] = transformations[current] @ transform
                calculated.add(j)
                to_process.append(j)
                
            elif j == current and i not in calculated:
                # Calculate transformation from i to j (current) 
                transform = _compute_marker_transformation(common_markers, from_proj=i, to_proj=j)
                # Chain with current's transformation
                transformations[i] = transformations[current] @ transform
                calculated.add(i)
                to_process.append(i)
    
    return transformations

def _compute_marker_transformation(common_markers: Dict[int, Tuple[np.ndarray, np.ndarray]], 
                                 from_proj: int, to_proj: int) -> np.ndarray:
    """Compute transformation matrix using marker correspondences."""
    
    # Extract corresponding point clouds
    source_points = []
    target_points = []
    
    for marker_id, (corners_A, corners_B) in common_markers.items():
        source_points.append(corners_A)  # from_proj corners
        target_points.append(corners_B)  # to_proj corners
    
    # Flatten corner arrays into point clouds
    source_cloud = np.vstack(source_points)  # Nx3
    target_cloud = np.vstack(target_points)  # Nx3
    
    # Compute transformation from source to target
    transform = get_transformation_between_clouds(source_cloud, target_cloud)
    
    return transform