import os
import numpy as np
import cv2
from aruco_estimator.colmap.read_write_model import *

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
    """Project 3D points to image coordinates."""
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

    # Project points
    img_points, _ = cv2.projectPoints(points, rvec, tvec, K, dist_coeffs)
    return img_points.reshape(-1, 2)

def draw_axes(img, cam_params, rvec, tvec, length=1):
    """Draw 3D coordinate axes on image."""
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

    # Define axis points in 3D
    axis_points = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]])
    
    # Project points
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist_coeffs)
    img_points = img_points.reshape(-1, 2)
    
    # Draw axes
    origin = tuple(map(int, img_points[0]))
    point_x = tuple(map(int, img_points[1]))
    point_y = tuple(map(int, img_points[2]))
    point_z = tuple(map(int, img_points[3]))
    
    img = cv2.line(img, origin, point_x, (0,0,255), 22)  # X axis: Red
    img = cv2.line(img, origin, point_y, (0,255,0), 22)  # Y axis: Green
    img = cv2.line(img, origin, point_z, (255,0,0), 22)  # Z axis: Blue
    
    return img

def normalize_coordinates(points, img_shape):
    """Normalize coordinates to [0,1] range."""
    h, w = img_shape[:2]
    points[:, 0] = points[:, 0] / w
    points[:, 1] = points[:, 1] / h
    return points

def create_label_content(projected_points, img_shape, class_id=0, visibility=2):
    """Create label content in YOLO format with keypoints."""
    # Normalize point coordinates
    norm_points = normalize_coordinates(projected_points.copy(), img_shape)
    
    # Calculate bounding box from points
    x_min, y_min = np.min(norm_points, axis=0)
    x_max, y_max = np.max(norm_points, axis=0)
    
    # Calculate center and dimensions
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    # Format label line
    label_parts = [str(class_id), f"{x_center:.6f}", f"{y_center:.6f}", 
                   f"{width:.6f}", f"{height:.6f}"]
    
    # Add keypoints
    for x, y in norm_points:
        label_parts.extend([f"{x:.6f}", f"{y:.6f}", str(visibility)])
    
    return " ".join(label_parts)

if __name__ == "__main__":
    # Set paths
    base_dir = "/Users/walkenz1/Projects/aruco-estimator/data/1_15_25"
    camera_dir = "C1"
    colmap_path = os.path.join(base_dir, camera_dir, "colmap", "sparse")
    
    # Create dataset structure
    dataset_root = os.path.join(base_dir, camera_dir, "dataset")
    images_dir = os.path.join(dataset_root, "images")
    labels_dir = os.path.join(dataset_root, "labels")
    
    # Create directories
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Read COLMAP model
    cameras, images, points3D = read_model(colmap_path)
    
    # Process all images
    for image_id, image in images.items():
        print(f"Processing {image.name}...")
        
        # Load image
        img_path = os.path.join(base_dir, camera_dir, "images", image.name)
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
        
        # Create and project dense grid of points
        grid_points = create_dense_grid(min_coords=(-2,-2,-2), max_coords=(2,2,2), num_points=4)
        colors = color_points_by_xyz(grid_points)
        projected_points = project_points(grid_points, cam_params, rvec, tvec)
        
        # Draw grid points
        for point, color in zip(projected_points, colors):
            x, y = map(int, point)
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (x, y), 3, color.tolist(), -1)
        
        # Draw coordinate axes
        img = draw_axes(img, cam_params, rvec, tvec, length=2)
        
        # Read and project key positions
        key_positions_path = os.path.join(base_dir, camera_dir, "key_positions.txt")
        if os.path.exists(key_positions_path):
            key_points = read_key_positions(key_positions_path)
            projected_key_points = project_points(key_points, cam_params, rvec, tvec)
            
            # Create label content
            label_content = create_label_content(projected_key_points, img.shape)
            
            # Save label
            label_path = os.path.join(labels_dir, f"{os.path.splitext(image.name)[0]}.txt")
            with open(label_path, 'w') as f:
                f.write(label_content)
            
            # Draw keypoints for visualization
            for point in projected_key_points:
                x, y = map(int, point)
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    cv2.circle(img, (x, y), 15, (0, 255, 255), 3)
        
        # Save image
        output_path = os.path.join(images_dir, image.name)
        cv2.imwrite(output_path, img)
        print(f"Saved to: {output_path}")
