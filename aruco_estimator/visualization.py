from typing import Dict, List, Optional

import cv2
import numpy as np
import open3d as o3d

from aruco_estimator.sfm.common import SfmProjectBase


class VisualizationModel:
    """3D visualization model that can display multiple projects/data sources."""
    
    def __init__(self):
        """Initialize empty visualization model."""
        self.__vis = None

    def create_window(self):
        """Create the visualization window."""
        self.__vis = o3d.visualization.Visualizer()
        self.__vis.create_window()

    def add_project(self, project: SfmProjectBase, 
                   points_config: Optional[Dict] = None,
                   cameras_config: Optional[Dict] = None,
                   enable_points: bool = True,
                   enable_cameras: bool = True):
        """
        Add a project to the visualization with specific configuration.
        
        Args:
            project: SfM project instance
            points_config: Configuration for point cloud visualization
            cameras_config: Configuration for camera visualization  
            enable_points: Whether to add points from this project
            enable_cameras: Whether to add cameras from this project
        """
        if enable_points and project.points3D:
            self._add_project_points(project, points_config or {})
        
        if enable_cameras and project.cameras and project.images:
            self._add_project_cameras(project, cameras_config or {})

    def _add_project_points(self, project: SfmProjectBase, config: Dict):
        """Add 3D points from a project with given configuration."""
        # Default configuration
        default_config = {
            "min_track_len": 3,
            "remove_statistical_outlier": True,
            "color": None
        }
        default_config.update(config)
        
        pcd = o3d.geometry.PointCloud()

        xyz = []
        rgb = []
        for point3D in project.points3D.values():
            track_len = len(point3D.point2D_idxs)
            if track_len < default_config["min_track_len"]:
                continue
            xyz.append(point3D.xyz)
            if default_config["color"] is None:
                rgb.append(point3D.rgb / 255)

        pcd.points = o3d.utility.Vector3dVector(xyz)
        if default_config["color"] is not None:
            pcd.paint_uniform_color(default_config["color"])
        else:
            pcd.colors = o3d.utility.Vector3dVector(rgb)

        # Remove obvious outliers
        if default_config["remove_statistical_outlier"]:
            [pcd, _] = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )

        self.__vis.add_geometry(pcd)
        self.__vis.poll_events()
        self.__vis.update_renderer()

    def _add_project_cameras(self, project: SfmProjectBase, config: Dict):
        """Add cameras from a project with given configuration."""
        # Default configuration
        default_config = {
            "scale": 1.0,
            "color": [0.8, 0.2, 0.8],
            "show_images": True,
            "image_alpha": 0.8
        }
        default_config.update(config)
        
        for img in project.images.values():
            # Get camera parameters
            cam = project.cameras[img.camera_id]
            
            # Get extrinsics matrix (camera-to-world)
            extrinsics = img.world_extrinsics
            
            # Get intrinsics matrix
            K = cam.K
            
            # Load the actual image if requested
            image = None
            if default_config["show_images"]:
                image = project.load_image_by_id(img.id)
                if image is not None:
                    # Convert BGR to RGB for Open3D
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create camera viewport with image texture
            line_set, spheres, mesh = draw_camera_viewport(
                extrinsics=extrinsics,
                intrinsics=K,
                image=image,
                scale=default_config["scale"]
            )
            
            # Add all geometries
            if mesh is not None and len(mesh.vertices) > 0:
                self.__vis.add_geometry(mesh)
            self.__vis.add_geometry(line_set)
            # spheres is a list from create_sphere_mesh
            for sphere in spheres:
                self.__vis.add_geometry(sphere)

    def add_coordinate_frame(self, size=1.0, transform=None):
        """Add coordinate frame visualization.
        
        Args:
            size: Size of the coordinate frame
            transform: Optional 4x4 transformation matrix to apply
        """
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        if transform is not None:
            frame.transform(transform)
        self.__vis.add_geometry(frame)

    def add_aruco_marker(self, corners, color=[1, 0, 1], corner_size=0.10):
        """Add ArUco marker visualization.
        
        Args:
            corners: Array of 4 corner points
            color: RGB color for marker edges
            corner_size: Size of corner spheres
        """
        # Create ArUco marker visualization
        aruco_lines = o3d.geometry.LineSet()
        aruco_lines.points = o3d.utility.Vector3dVector(corners)
        aruco_lines.lines = o3d.utility.Vector2iVector([[0,1], [1,2], [2,3], [3,0]])
        aruco_lines.colors = o3d.utility.Vector3dVector([color for _ in range(4)])
        
        # Add lines
        self.__vis.add_geometry(aruco_lines)
        
        # Add corner spheres instead of points for better visibility
        corner_color = [1, 1, 0]  # Yellow corners
        for corner in corners:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=corner_size)
            sphere.translate(corner)
            sphere.paint_uniform_color(corner_color)
            self.__vis.add_geometry(sphere)

    def add_aruco_markers(self, markers_dict: Dict[int, np.ndarray], 
                         colors: Optional[List[List[float]]] = None,
                         corner_size: float = 0.10):
        """Add multiple ArUco markers with different colors.
        
        Args:
            markers_dict: Dictionary of {marker_id: corners} 
            colors: List of RGB colors, cycles if fewer colors than markers
            corner_size: Size of corner spheres
        """
        if colors is None:
            colors = [
                [1, 0, 0],  # Red
                [0, 1, 0],  # Green  
                [0, 0, 1],  # Blue
                [1, 1, 0],  # Yellow
                [1, 0, 1],  # Magenta
                [0, 1, 1],  # Cyan
                [0.5, 0.5, 0],  # Olive
                [0.5, 0, 0.5],  # Purple
                [0, 0.5, 0.5],  # Teal
                [0.7, 0.3, 0.3],  # Brown
            ]
        
        for i, (marker_id, corners) in enumerate(markers_dict.items()):
            color = colors[i % len(colors)]
            self.add_aruco_marker(corners, color=color, corner_size=corner_size)

    def show(self):
        """Display the visualization."""
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()


def draw_camera_viewport(extrinsics, intrinsics, image=None, scale=1.0, 
                        image_width=None, image_height=None, color=[0.8, 0.2, 0.8]):
    """
    Create camera viewport visualization exactly like colmap_wrapper.visualization.draw_camera_viewport
    
    Args:
        extrinsics: 4x4 camera-to-world transformation matrix
        intrinsics: 3x3 camera intrinsics matrix K
        image: Camera image (BGR format)
        scale: Scale factor for camera visualization
        image_width: Image width in pixels
        image_height: Image height in pixels
        color: Fallback color if no image provided
    
    Returns:
        tuple: (line_set, spheres, mesh) representing camera frustum, points, and image plane
    """
    
    # Extrinsic parameters - extract R and t from 4x4 matrix
    R, t = extrinsics[:3, :3], extrinsics[:3, 3]
    
    # Intrinsic parameters
    fx, fy, cx, cy = (
        intrinsics[0, 0],
        intrinsics[1, 1], 
        intrinsics[0, 2],
        intrinsics[1, 2],
    )
    
    # Normalize to 1 (this is the key difference from my previous implementation)
    max_norm = max(fx, fy, cx, cy)
    
    # Define frustum points exactly as in the working version
    points = [
        t,
        t + (np.asarray([cx, cy, fx]) * scale) / max_norm @ R.T,
        t + (np.asarray([cx, -cy, fx]) * scale) / max_norm @ R.T,
        t + (np.asarray([-cx, -cy, fx]) * scale) / max_norm @ R.T,
        t + (np.asarray([-cx, cy, fx]) * scale) / max_norm @ R.T,
    ]
    
    # Define frustum lines
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    line_set = generate_line_set(points=points, lines=lines, color=[1, 0, 0])
    
    # Create sphere at camera center
    sphere = create_sphere_mesh(t=t, color=[1, 0, 0], radius=0.01)
    
    # Fill image plane/mesh with image as texture
    if isinstance(image, np.ndarray):
        # Create Point Cloud and assign a normal vector pointing in the opposite direction of the viewing normal
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points[1:]))
        normal_vec = -(np.asarray([0, 0, fx]) @ R.T)
        pcd.normals = o3d.utility.Vector3dVector(
            np.tile(normal_vec, (pcd.points.__len__(), 1))
        )
        
        # Create image plane with image as texture
        plane = o3d.geometry.TriangleMesh()
        plane.vertices = pcd.points
        plane.triangles = o3d.utility.Vector3iVector(np.asarray([[0, 1, 3], [1, 2, 3]]))
        plane.compute_vertex_normals()
        v_uv = np.asarray([[1, 1], [1, 0], [0, 1], [1, 0], [0, 0], [0, 1]])
        plane.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
        plane.triangle_material_ids = o3d.utility.IntVector([0] * 2)
        plane.textures = [o3d.geometry.Image(image)]
    else:
        plane = o3d.geometry.TriangleMesh()
    
    return line_set, sphere, plane


def generate_line_set(points: list, lines: list, color: list) -> o3d.geometry.LineSet:
    """
    Generates a line set of parsed points, with uniform color.

    :param points: points of lines
    :param lines: list of connections
    :param color: rgb color ranging between 0 and 1.
    :return:
    """
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def create_sphere_mesh(t: np.ndarray, color: list, radius: float) -> list:
    """
    Creates a sphere mesh, is translated to a parsed 3D coordinate and has uniform color

    @param t: 3D Coordinate. Either 1 coordinate ore multiple
    @param color: rgb color ranging between 0 and 1.
    @param radius: radius of the sphere
    :return:
    """
    # Only one point
    if t.shape.__len__() == 1:
        t = np.expand_dims(t, axis=0)
        color = [color]

    sphere_list = []

    for p, c in zip(t, color):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(p)
        sphere.paint_uniform_color(np.asarray(c))
        sphere.compute_triangle_normals()
        sphere_list.append(sphere)

    return sphere_list