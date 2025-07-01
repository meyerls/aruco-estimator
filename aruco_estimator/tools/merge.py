import logging
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from aruco_estimator.utils import (
    get_transformation_between_clouds,
)
from aruco_estimator.sfm.common import SfmProjectBase
from aruco_estimator.sfm.colmap import COLMAPProject


# region Project Merging Tools
def merge_projects(
    projects: List["SfmProjectBase"],
    output_path: str,
    images_path: Optional[str] = None,
) -> "COLMAPProject":
    """
    Merge multiple SfM projects with optional physical image copying.
    Returns a new COLMAPProject pointing to merged data.

    :param projects: List of SfM projects to merge
    :param output_path: Directory for merged reconstruction data
    :param images_path: Directory to copy images to (None = no copying)
    :return: New COLMAPProject instance
    """
    if len(projects) < 2:
        raise ValueError("Need at least 2 projects to merge")

    merged_cameras = {}
    merged_images = {}
    merged_points3D = {}
    merged_markers = {}

    # Calculate ID assignments for all projects (using offset strategy)
    id_assignments = _calculate_all_id_assignments(projects)

    # Setup image copying if requested
    image_name_mapping = {}  # {old_path: new_name}
    image_counter = 1

    if images_path:
        # Create images directory
        Path(images_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Created images directory: {images_path}")

    # Merge each project
    for idx, project in enumerate(projects):
        assignment = id_assignments[idx]

        # Merge cameras
        for old_id, new_id in assignment["cameras"].items():
            camera = project.cameras[old_id]
            merged_cameras[new_id] = camera._replace(id=new_id)

        # Merge images
        for old_id, new_id in assignment["images"].items():
            image = project.images[old_id]
            new_camera_id = assignment["cameras"][image.camera_id]

            # Handle image copying and naming
            if images_path:
                # Copy image to new location with sequential naming
                old_image_path = os.path.join(project.images_path, image.name)
                file_ext = Path(image.name).suffix
                new_image_name = f"img_{image_counter:06d}{file_ext}"
                new_image_path = os.path.join(images_path, new_image_name)

                # Copy the image
                try:
                    shutil.copy2(old_image_path, new_image_path)
                    logging.debug(f"Copied {old_image_path} -> {new_image_path}")
                except Exception as e:
                    logging.warning(f"Failed to copy {old_image_path}: {e}")
                    new_image_name = image.name  # Keep original name if copy fails

                image_name_mapping[old_image_path] = new_image_name
                image_counter += 1
            else:
                # Keep original image name/path
                new_image_name = image.name

            # Update point3D_ids references
            new_point3D_ids = None
            if image.point3D_ids is not None:
                new_point3D_ids = []
                for old_pt_id in image.point3D_ids:
                    if old_pt_id == -1:
                        new_point3D_ids.append(-1)
                    elif old_pt_id in assignment["points"]:
                        new_point3D_ids.append(assignment["points"][old_pt_id])
                    else:
                        new_point3D_ids.append(-1)
                new_point3D_ids = np.array(new_point3D_ids)

            merged_images[new_id] = image._replace(
                id=new_id,
                camera_id=new_camera_id,
                point3D_ids=new_point3D_ids,
                name=new_image_name,
            )

        # Merge 3D points
        for old_id, new_id in assignment["points"].items():
            point = project.points3D[old_id]
            merged_points3D[new_id] = point._replace(id=new_id)

        # Merge markers
        if hasattr(project, "_markers") and project._markers:
            for dict_type, markers in project.markers.items():
                if dict_type not in merged_markers:
                    merged_markers[dict_type] = {}

                for marker_id, marker in markers.items():
                    if marker_id in merged_markers[dict_type]:
                        # Combine observations for same ArUco marker
                        existing = merged_markers[dict_type][marker_id]
                        combined_image_ids = list(existing.image_ids) + list(
                            marker.image_ids
                        )
                        combined_point2D_idxs = list(existing.point2D_idxs) + list(
                            marker.point2D_idxs
                        )
                        avg_xyz = (existing.xyz + marker.xyz) / 2
                        avg_corners = (existing.corners_3d + marker.corners_3d) / 2

                        merged_markers[dict_type][marker_id] = existing._replace(
                            xyz=avg_xyz,
                            corners_3d=avg_corners,
                            image_ids=combined_image_ids,
                            point2D_idxs=combined_point2D_idxs,
                        )
                    else:
                        merged_markers[dict_type][marker_id] = marker

    # Create output directories
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Create a new COLMAPProject instance
    merged_project = COLMAPProject.__new__(COLMAPProject)
    merged_project._project_path = output_path
    merged_project._cameras = merged_cameras
    merged_project._images = merged_images
    merged_project._points3D = merged_points3D
    merged_project._markers = merged_markers
    merged_project.images_path = images_path if images_path else projects[0].images_path
    merged_project.sparse_folder = "."  # Since output_path is the sparse folder
    merged_project.dense_path = None
    merged_project.dense_point_cloud = None

    logging.info(
        f"Merged {len(projects)} projects: {len(merged_cameras)} cameras, "
        f"{len(merged_images)} images, {len(merged_points3D)} points"
    )

    if images_path:
        logging.info(f"Copied {len(image_name_mapping)} images to {images_path}")

    return merged_project


def _calculate_all_id_assignments(projects: List["SfmProjectBase"]) -> List[Dict]:
    """Calculate new ID assignments for all projects to avoid conflicts."""

    assignments = []

    # Sequential assignment (offset strategy)
    next_camera_id = 1
    next_image_id = 1
    next_point_id = 1

    for project in projects:
        assignment = {"cameras": {}, "images": {}, "points": {}}

        for old_id in sorted(project.cameras.keys()):
            assignment["cameras"][old_id] = next_camera_id
            next_camera_id += 1

        for old_id in sorted(project.images.keys()):
            assignment["images"][old_id] = next_image_id
            next_image_id += 1

        for old_id in sorted(project.points3D.keys()):
            assignment["points"][old_id] = next_point_id
            next_point_id += 1

        assignments.append(assignment)

    return assignments


# endregion


def align_projects(
    projects: List["SfmProjectBase"],
    target_id: int = 0,
) -> List["SfmProjectBase"]:
    """
    Align multiple projects using common ArUco markers.

    :param projects: List of SfM projects to align
    :param target_id: Index of target project (root coordinate system)
    :return: List of aligned projects
    """
    if target_id >= len(projects):
        raise ValueError(
            f"target_id {target_id} out of range for {len(projects)} projects"
        )

    # Find marker links between all project pairs
    links = {}
    for i in range(len(projects)):
        for j in range(len(projects)):
            if i == j:
                continue

        common_markers = _find_common_markers(projects[i], projects[j])
        links[(i, j)] = common_markers
        logging.info(
            f"Found {len(common_markers)} common markers between projects {i} and {j}"
        )

    if not links:
        raise ValueError("No projects have sufficient common markers for alignment")

    # Check connectivity - ensure all projects can be reached from target
    connected_projects = _find_connected_projects(links, target_id, len(projects))
    disconnected = set(range(len(projects))) - connected_projects
    if disconnected:
        logging.warning(
            f"Projects {disconnected} cannot be connected to target project {target_id}"
        )

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


def _find_common_markers(
    proj_A: "SfmProjectBase", proj_B: "SfmProjectBase"
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Find common ArUco markers between two projects."""
    common_markers = {}

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


def _find_connected_projects(
    links: Dict[Tuple[int, int], Dict], target_id: int, num_projects: int
) -> set:
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


def _calculate_transformations(
    projects: List["SfmProjectBase"], links: Dict[Tuple[int, int], Dict], target_id: int
) -> Dict[int, np.ndarray]:
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
                transform = _compute_marker_transformation(
                    common_markers, from_proj=j, to_proj=i
                )
                # Chain with current's transformation
                transformations[j] = transformations[current] @ transform
                calculated.add(j)
                to_process.append(j)

            elif j == current and i not in calculated:
                # Calculate transformation from i to j (current)
                transform = _compute_marker_transformation(
                    common_markers, from_proj=i, to_proj=j
                )
                # Chain with current's transformation
                transformations[i] = transformations[current] @ transform
                calculated.add(i)
                to_process.append(i)

    return transformations


def _compute_marker_transformation(
    common_markers: Dict[int, Tuple[np.ndarray, np.ndarray]],
    from_proj: int,
    to_proj: int,
) -> np.ndarray:
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
