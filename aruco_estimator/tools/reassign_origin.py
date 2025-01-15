# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
import os
import numpy as np
from aruco_estimator.colmap.read_write_model import (
    read_model,
    write_model,
    Image,
    Point3D,
    qvec2rotmat,
    rotmat2qvec
)
import open3d
from aruco_estimator.colmap.visualize_model import Model
from scipy.spatial.transform import Rotation
from colmap_wrapper.colmap import COLMAP, generate_colmap_sparse_pc
from aruco_estimator.localizers import ArucoLocalizer

def get_normalization_transform(aruco_corners_3d: np.ndarray) -> np.ndarray:
    """Calculate transformation matrix to normalize coordinates to ArUco marker plane."""
    if len(aruco_corners_3d) != 4:
        raise ValueError(f"Expected 4 ArUco corners, got {len(aruco_corners_3d)}")
    
    # Calculate ArUco center
    aruco_center = np.mean(aruco_corners_3d, axis=0)
    
    # Find the shortest and longest edges to determine marker orientation
    edges = []
    for i in range(4):
        next_i = (i + 1) % 4
        edge = aruco_corners_3d[next_i] - aruco_corners_3d[i]
        edges.append((np.linalg.norm(edge), edge, i))
    edges.sort(key=lambda x: x[0])  # Sort by edge length
    
    # Use shortest edge for y direction (typically marker height)
    # and its perpendicular for x direction (marker width)
    y_vec = edges[0][1]  # Shortest edge
    y_vec = y_vec / np.linalg.norm(y_vec)
    
    # Find the edge most perpendicular to y_vec
    perp_scores = []
    for _, edge, idx in edges[1:]:  # Skip shortest edge
        edge_norm = edge / np.linalg.norm(edge)
        # Score based on how close to perpendicular (dot product near 0)
        score = abs(np.dot(y_vec, edge_norm))
        perp_scores.append((score, edge_norm, idx))
    _, x_vec, _ = min(perp_scores, key=lambda x: x[0])
    
    # Calculate z-axis ensuring right-handed coordinate system
    z_vec = np.cross(x_vec, y_vec)
    z_vec = z_vec / np.linalg.norm(z_vec)
    
    # Ensure z-axis points upward
    if z_vec[2] < 0:
        z_vec = -z_vec
        x_vec = -x_vec  # Flip x to maintain right-handed system
    
    # Ensure y is exactly perpendicular
    y_vec = np.cross(z_vec, x_vec)
    y_vec = y_vec / np.linalg.norm(y_vec)
    
    # Create rotation matrix
    rotation = np.stack([x_vec, y_vec, z_vec], axis=0)
    
    # Create full transform
    transform = np.eye(4)
    transform[:3, :3] = rotation.T
    transform[:3, 3] = -rotation.T @ aruco_center
    
    return transform

def normalize_poses_and_points(cameras, images, points3D, transform: np.ndarray):
    """Apply normalization transform to camera poses and 3D points"""
    # Transform camera poses
    transformed_images = {}
    for image_id, image in images.items():
        # Get current rotation and translation
        R = qvec2rotmat(image.qvec)
        t = image.tvec

        # For camera poses, we need to apply the inverse transformation
        # R_new = R @ R_transform.T
        # t_new = t - R @ R_transform.T @ t_transform
        new_R = R @ transform[:3, :3].T
        new_t = t - R @ transform[:3, :3].T @ transform[:3, 3]

        # Create new image with transformed pose
        transformed_images[image_id] = Image(
            id=image.id,
            qvec=rotmat2qvec(new_R),
            tvec=new_t,
            camera_id=image.camera_id,
            name=image.name,
            xys=image.xys,
            point3D_ids=image.point3D_ids
        )
    
    # Transform 3D points
    transformed_points3D = {}
    for point3D_id, point3D in points3D.items():
        new_xyz = transform[:3, :3] @ point3D.xyz + transform[:3, 3]
        transformed_points3D[point3D_id] = Point3D(
            id=point3D.id,
            xyz=new_xyz,
            rgb=point3D.rgb,
            error=point3D.error,
            image_ids=point3D.image_ids,
            point2D_idxs=point3D.point2D_idxs
        )
    
    return cameras, transformed_images, transformed_points3D

def save_normalized_data(cameras, images, points3D, output_path: Path) -> None:
    """Save normalized poses and points using COLMAP structure"""
    # Create normalized/sparse directory
    output_dir = output_path / "normalized" / "sparse"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write transformed data
    write_model(cameras, images, points3D, str(output_dir), ".txt")

def main():
    parser = argparse.ArgumentParser(description='Normalize COLMAP poses relative to ArUco marker')
    parser.add_argument('--colmap_project', type=str, required=True,
                       help='Path to COLMAP project containing images.txt and points3D.txt')
    parser.add_argument('--aruco_size', type=float, help='Size of the aruco marker in meter.', default=0.2)
    args = parser.parse_args()
    
    project_path = Path(args.colmap_project)
    logging.basicConfig(level=logging.INFO)
    
    # Load COLMAP data
    logging.info("Loading COLMAP data...")
    sparse_dir = os.path.join(project_path, "sparse")
    cameras, images, points3D = read_model(sparse_dir)
    
    # Use COLMAP project only for ArUco detection
    logging.info("Detecting ArUco markers...")
    project = COLMAP(project_path=str(project_path))
    aruco_localizer = ArucoLocalizer(
        photogrammetry_software=project,
        aruco_size=args.aruco_size,
    )
    aruco_distance, aruco_corners_3d = aruco_localizer.run()
    logging.info(f"ArUco 3d points: {aruco_corners_3d}")
    logging.info(f"ArUco marker distance: {aruco_distance}")
    
    # Calculate normalization transform
    transform = get_normalization_transform(aruco_corners_3d)
    
    # Apply normalization to loaded data
    logging.info("Normalizing poses and 3D points...")
    cameras_norm, images_norm, points3D_norm = normalize_poses_and_points(cameras, images, points3D, transform)
    
    # Create visualization model
    model = Model()
    model.create_window()
    
    # Add point clouds
    model.points3D = points3D
    model.add_points(color=[0.7, 0.7, 0.7])  # Gray for original points
    
    model.points3D = points3D_norm
    model.add_points(color=[0, 0.7, 1])  # Light blue for transformed points
    
    # Add coordinate frames
    model.add_coordinate_frame(size=2.0)  # Original coordinate frame
    model.add_coordinate_frame(size=1.0, transform=transform)  # Transformed coordinate frame
    
    # Add ArUco markers
    model.add_aruco_marker(aruco_corners_3d, color=[1, 0, 1])  # Magenta for original marker
    
    # Transform ArUco corners to new coordinate system
    transformed_corners = np.array([
        transform[:3, :3] @ corner + transform[:3, 3] 
        for corner in aruco_corners_3d
    ])
    model.add_aruco_marker(transformed_corners, color=[0, 1, 1])  # Cyan for transformed marker
    
    # Log transformed corners for verification
    logging.info("Transformed ArUco corners:")
    for i, corner in enumerate(transformed_corners):
        logging.info(f"Corner {i}: {corner}")
    
    # Add cameras
    model.cameras = cameras
    model.images = images
    model.add_cameras(scale=0.25, color=[0.7, 0.7, 0.7])  # Dark yellow for original cameras
    
    model.cameras = cameras_norm
    model.images = images_norm
    model.add_cameras(scale=0.25, color=[1, 0.5, 0])  # Orange for transformed cameras
    
    # Show visualization
    model.show()
    
    # Save transformed data
    logging.info("Saving normalized data...")
    save_normalized_data(cameras_norm, images_norm, points3D_norm, project_path)
    
    logging.info("Done! Normalized data saved to normalized/sparse/")

    logging.info("Done! Normalized data saved to normalized/sparse/")

if __name__ == '__main__':
    main()
