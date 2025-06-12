import os
import cv2
import numpy as np

from aruco_estimator.colmap.read_write_model import read_model


def read_key_positions(filepath):
    """Read key positions from file."""
    points = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            x, z, y = map(float, line.strip().split())
            points.append([x, -y, z])
    return np.array(points)

def create_dense_grid(min_coords=(-1,-1,-1), max_coords=(1,1,1), num_points=20):
    """Create a dense grid of 3D points."""
    x = np.linspace(min_coords[0], max_coords[0], num_points)
    y = np.linspace(min_coords[1], max_coords[1], num_points)
    z = np.linspace(min_coords[2], max_coords[2], num_points)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Reshape to Nx3 array of points
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    return points

def color_points_by_xyz(points):
    """Color points based on their XYZ coordinates."""
    # Normalize coordinates to [0,1] range for each axis
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    normalized = (points - min_coords) / (max_coords - min_coords)
    
    # Use normalized coordinates directly as RGB values
    colors = (normalized * 255).astype(np.uint8)
    return colors

def project_points(points, cam_params, rvec, tvec):
    """
    Project 3D points to image coordinates.
    Returns projected points and a mask indicating which points are in front of the camera.
    """
    # Camera matrix
    if cam_params.model == "SIMPLE_PINHOLE":
        fx = fy = cam_params.params[0]
        cx, cy = cam_params.params[1:]
        dist_coeffs = np.zeros(4)
    elif cam_params.model == "PINHOLE":
        fx, fy = cam_params.params[0:2]
        cx, cy = cam_params.params[2:]
        dist_coeffs = np.zeros(4)
    elif cam_params.model == "SIMPLE_RADIAL":
        fx = fy = cam_params.params[0]
        cx, cy = cam_params.params[1:3]
        k = cam_params.params[3]
        dist_coeffs = np.array([k, 0, 0, 0])  # k1, k2, p1, p2
    else:
        raise ValueError(f"Unsupported camera model: {cam_params.model}")
    
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    # Convert points to camera coordinate system
    R, _ = cv2.Rodrigues(rvec)
    points_cam = (R @ points.T).T + tvec.reshape(1, 3)
    
    # Check which points are in front of the camera (Z > 0)
    in_front = points_cam[:, 2] > 0
    
    # Project points
    img_points, _ = cv2.projectPoints(points, rvec, tvec, K, dist_coeffs)
    img_points = img_points.reshape(-1, 2)
    
    # Set points behind camera to NaN
    img_points[~in_front] = np.nan
    
    return img_points, in_front

def draw_axes(img, cam_params, rvec, tvec, length=1):
    """Draw 3D coordinate axes on image."""
    # Define axis points in 3D
    axis_points = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]])
    
    # Project points using existing function
    img_points, in_front = project_points(axis_points, cam_params, rvec, tvec)
    
    # Draw axes only if points are in front of camera
    origin = tuple(map(int, img_points[0]))
    for i, (point, visible) in enumerate(zip(img_points[1:], in_front[1:]), 1):
        if visible and not np.isnan(point).any():
            point = tuple(map(int, point))
            color = [(0,0,255), (0,255,0), (255,0,0)][i-1]  # Red, Green, Blue for X, Y, Z
            img = cv2.line(img, origin, point, color, 22)
    
    return img

def normalize_coordinates(points, img_shape):
    """Normalize coordinates to [0,1] range."""
    h, w = img_shape[:2]
    points[:, 0] = points[:, 0] / w
    points[:, 1] = points[:, 1] / h
    return points

def create_label_content(projected_points, img_shape, in_front, class_id=0):
    """Create label content in YOLO format with keypoints."""
    if len(projected_points) == 0:
        return None  # Return None if no points
    
    h, w = img_shape[:2]
    
    # Check which points are within image bounds
    in_bounds = np.logical_and.reduce([
        projected_points[:, 0] >= 0,
        projected_points[:, 0] < w,
        projected_points[:, 1] >= 0,
        projected_points[:, 1] < h
    ])
    
    # Point is visible if it's both in front of camera and within image bounds
    is_visible = np.logical_and(in_front, in_bounds)
    
    # Get valid points for bounding box calculation (not NaN)
    valid_mask = ~np.isnan(projected_points).any(axis=1)
    valid_points = projected_points[valid_mask]
    
    if len(valid_points) == 0:
        # If all points are NaN, use center of image for bounding box
        x_center, y_center = 0.5, 0.5
        width, height = 0.0, 0.0
    else:
        # Normalize valid points and calculate bounding box
        norm_valid_points = normalize_coordinates(valid_points.copy(), img_shape)
        x_min, y_min = np.min(norm_valid_points, axis=0)
        x_max, y_max = np.max(norm_valid_points, axis=0)
        
        # Calculate center and dimensions
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
    
    # Format label line
    label_parts = [str(class_id), f"{x_center:.6f}", f"{y_center:.6f}", 
                   f"{width:.6f}", f"{height:.6f}"]
    
    # Add keypoints
    for i, ((x, y), visible) in enumerate(zip(projected_points, is_visible)):
        if np.isnan(x) or np.isnan(y):
            # Set NaN points to 0,0 with visibility 0
            label_parts.extend(["0.000000", "0.000000", "0"])
        else:
            # Normalize and add valid points
            norm_x = x / img_shape[1]
            norm_y = y / img_shape[0]
            visibility = 2 if visible else 0
            label_parts.extend([f"{norm_x:.6f}", f"{norm_y:.6f}", str(visibility)])
    
    return " ".join(label_parts)

def reverse_project(colmap_path, images_path, output_dir, key_positions_path=None, 
                   grid_min=(-2,-2,-2), grid_max=(2,2,2), grid_points=4,
                   skip_copy=False, draw_visualization=False):
    """
    Project 3D points onto images and create dataset.
    
    Args:
        colmap_path: Path to COLMAP sparse reconstruction directory
        images_path: Path to directory containing source images
        output_dir: Path to output directory for processed images and labels
        key_positions_path: Optional path to key positions text file
        grid_min: Minimum coordinates for grid (x,y,z)
        grid_max: Maximum coordinates for grid (x,y,z) 
        grid_points: Number of points per dimension in grid
        skip_copy: If True, don't save visualization images
        draw_visualization: If True, draw grid points and axes on visualization images
    """
    # Create dataset structure
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    if draw_visualization and not skip_copy:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    
    # Read COLMAP model
    cameras, images, points3D = read_model(colmap_path)
    
    # Process all images
    for image_id, image in images.items():
        print(f"Processing {image.name}...")
        
        # Load image
        img_path = os.path.join(images_path, image.name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {img_path}")
            continue
        
        # Get camera parameters
        cam_params = cameras[image.camera_id]
        
        # Convert quaternion to rotation matrix
        R = image.qvec2rotmat()
        t = image.tvec
        
        # Convert to OpenCV convention
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3,1)
        
        # Create visualization image if needed
        vis_img = img.copy() if draw_visualization else None
        
        if draw_visualization:
            # Create and project dense grid of points
            points = create_dense_grid(min_coords=grid_min, max_coords=grid_max, num_points=grid_points)
            colors = color_points_by_xyz(points)
            projected_points, in_front = project_points(points, cam_params, rvec, tvec)
            
            # Draw grid points (only those in front of camera)
            for point, color, visible in zip(projected_points, colors, in_front):
                if visible and not np.isnan(point).any():
                    x, y = map(int, point)
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        cv2.circle(vis_img, (x, y), 3, color.tolist(), -1)
            
            # Draw coordinate axes
            vis_img = draw_axes(vis_img, cam_params, rvec, tvec, length=2)
        
        # Read and project key positions
        if key_positions_path and os.path.exists(key_positions_path):
            key_points = read_key_positions(key_positions_path)
            projected_key_points, key_in_front = project_points(key_points, cam_params, rvec, tvec)
            
            # Create label content with visibility information
            label_content = create_label_content(projected_key_points, img.shape, key_in_front)
            
            if label_content is not None:
                # Save label
                label_path = os.path.join(labels_dir, f"{os.path.splitext(image.name)[0]}.txt")
                with open(label_path, 'w') as f:
                    f.write(label_content)
            
            # Draw keypoints for visualization if enabled
            if draw_visualization:
                for point, visible in zip(projected_key_points, key_in_front):
                    if visible and not np.isnan(point).any():
                        x, y = map(int, point)
                        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                            cv2.circle(vis_img, (x, y), 15, (0, 255, 255), 3)
        
        # Save visualization if enabled and not skipping
        if draw_visualization and not skip_copy:
            output_path = os.path.join(vis_dir, f"vis_{image.name}")
            cv2.imwrite(output_path, vis_img)
            print(f"Saved visualization to: {output_path}")
