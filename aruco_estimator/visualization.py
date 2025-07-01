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

    def add_project(
        self,
        project: SfmProjectBase,
        points_config: Optional[Dict] = None,
        cameras_config: Optional[Dict] = None,
        markers_config: Optional[Dict] = None,
        enable_points: bool = True,
        enable_cameras: bool = True,
        enable_markers: bool = True,
    ):
        """
        Add a project to the visualization with specific configuration.

        Args:
            project: SfM project instance
            points_config: Configuration for point cloud visualization
            cameras_config: Configuration for camera visualization
            markers_config: Configuration for ArUco marker visualization
            enable_points: Whether to add points from this project
            enable_cameras: Whether to add cameras from this project
            enable_markers: Whether to add ArUco markers from this project
        """
        if enable_points and project.points3D:
            self._add_project_points(project, points_config or {})

        if enable_cameras and project.cameras and project.images:
            self._add_project_cameras(project, cameras_config or {})

        if enable_markers and hasattr(project, "markers") and project.markers:
            self._add_project_markers(project, markers_config or {})

    def _add_project_points(self, project: SfmProjectBase, config: Dict):
        """Add 3D points from a project with given configuration."""
        # Default configuration
        default_config = {"color": None}
        default_config.update(config)

        pcd = o3d.geometry.PointCloud()

        xyz = []
        rgb = []
        for point3D in project.points3D.values():
            xyz.append(point3D.xyz)
            if default_config["color"] is None:
                rgb.append(point3D.rgb / 255)

        pcd.points = o3d.utility.Vector3dVector(xyz)
        if default_config["color"] is not None:
            pcd.paint_uniform_color(default_config["color"])
        else:
            pcd.colors = o3d.utility.Vector3dVector(rgb)

        if project.dense_point_cloud is not None:
            self.__vis.add_geometry(project.dense_point_cloud)

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
            "image_alpha": 0.6,
        }
        default_config.update(config)

        for img in project.images.values():
            # Get camera parameters
            cam = project.cameras[img.camera_id]

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
                scale=default_config["scale"],
            )

            # Add all geometries
            if mesh is not None and len(mesh.vertices) > 0:
                self.__vis.add_geometry(mesh)
            self.__vis.add_geometry(line_set)
            # spheres is a list from create_sphere_mesh
            for sphere in spheres:
                self.__vis.add_geometry(sphere)

    def _add_project_markers(self, project: SfmProjectBase, config: Dict):
        """Add ArUco markers from a project with given configuration."""
        # Default configuration
        default_config = {
            "corner_size": 0.05,
            # "show_ids": True,
            "colors_by_dict": {
                # Default colors for different dictionary types
                "default": [1, 0, 1],  # Magenta
            },
            "dict_type_filter": None,  # None means show all, or list of dict_types to show
            "show_center": True,
            "center_size": 0.02,
            "center_color": [1, 1, 0],  # Yellow
            "edge_width": 2.0,
            "show_detection_lines": True,  # Show lines from cameras to markers (DEFAULT)
            "detection_line_color": [0.5, 0.0, 0.1],  # Gray lines
            "detection_line_alpha": 0.3,  # Line transparency (not fully supported in Open3D)
            "line_to_corners": True,  # If True, lines to corners; if False, lines to center
        }
        default_config.update(config)

        # Color cycle for different markers within same dictionary
        marker_colors = [
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

        color_idx = 0

        for dict_type, markers in project.markers.items():
            # Filter by dictionary type if specified
            if (
                default_config["dict_type_filter"] is not None
                and dict_type not in default_config["dict_type_filter"]
            ):
                continue

            for aruco_id, marker in markers.items():
                try:
                    # Get 3D corners for this marker
                    marker = project.markers[dict_type][aruco_id]
                    corners_3d = marker.corners_3d

                    # Use cycling colors for different markers
                    marker_color = marker_colors[color_idx % len(marker_colors)]
                    color_idx += 1

                    # Add marker visualization
                    self._add_single_marker(
                        corners_3d=corners_3d,
                        marker_id=aruco_id,
                        dict_type=dict_type,
                        color=marker_color,
                        config=default_config,
                    )

                    # Add detection lines if requested
                    if default_config["show_detection_lines"]:
                        self._add_detection_lines(
                            project=project,
                            marker=marker,
                            corners_3d=corners_3d,
                            config=default_config,
                        )

                except Exception as e:
                    print(
                        f"Warning: Could not visualize marker dict={dict_type}, id={aruco_id}: {e}"
                    )
                    continue

        self.__vis.poll_events()
        self.__vis.update_renderer()

    def _add_detection_lines(
        self, project: SfmProjectBase, marker, corners_3d: np.ndarray, config: Dict
    ):
        """Add lines from cameras that detected this marker to the marker's 3D position."""

        # Get camera centers for images that detected this marker
        camera_centers = []
        for image_id in marker.image_ids:
            if image_id in project.images:
                image = project.images[image_id]
                camera_center = image.get_camera_center()
                camera_centers.append(camera_center)

        if not camera_centers:
            return

        # Determine target points (corners or center)
        if config["line_to_corners"]:
            target_points = corners_3d  # Lines to all 4 corners
        else:
            target_points = [np.mean(corners_3d, axis=0)]  # Line to center only

        # Create lines from each camera center to target points
        all_points = []
        all_lines = []
        point_idx = 0

        for camera_center in camera_centers:
            for target_point in target_points:
                # Add camera center and target point
                all_points.extend([camera_center, target_point])

                # Add line connecting them
                all_lines.append([point_idx, point_idx + 1])
                point_idx += 2

        if all_points and all_lines:
            # Create line set
            detection_lines = o3d.geometry.LineSet()
            detection_lines.points = o3d.utility.Vector3dVector(all_points)
            detection_lines.lines = o3d.utility.Vector2iVector(all_lines)
            detection_lines.colors = o3d.utility.Vector3dVector(
                [config["detection_line_color"] for _ in range(len(all_lines))]
            )

            self.__vis.add_geometry(detection_lines)

    def _add_single_marker(
        self,
        corners_3d: np.ndarray,
        marker_id: int,
        dict_type: int,
        color: List[float],
        config: Dict,
    ):
        """Add a single ArUco marker to the visualization."""

        # Create marker edges
        aruco_lines = o3d.geometry.LineSet()
        aruco_lines.points = o3d.utility.Vector3dVector(corners_3d)
        aruco_lines.lines = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]])
        aruco_lines.colors = o3d.utility.Vector3dVector([color for _ in range(4)])

        # Make lines thicker if supported
        if hasattr(aruco_lines, "line_width"):
            aruco_lines.line_width = config["edge_width"]

        self.__vis.add_geometry(aruco_lines)

        # Add corner spheres for better visibility
        corner_color = [1, 1, 0]  # Yellow corners
        for corner in corners_3d:
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=config["corner_size"]
            )
            sphere.translate(corner)
            sphere.paint_uniform_color(corner_color)
            self.__vis.add_geometry(sphere)

        # Add center point if requested
        if config["show_center"]:
            center = np.mean(corners_3d, axis=0)
            center_sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=config["center_size"]
            )
            center_sphere.translate(center)
            center_sphere.paint_uniform_color(config["center_color"])
            self.__vis.add_geometry(center_sphere)

        # # Add text label if requested (note: Open3D text support is limited)
        # if config["show_ids"]:
        #     # Create a small coordinate frame at marker center to indicate ID
        #     # (Open3D doesn't have great text support, so this is a visual indicator)
        #     center = np.mean(corners_3d, axis=0)
        #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=config["corner_size"] * 2)
        #     frame.translate(center)
        #     self.__vis.add_geometry(frame)

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

    def add_detection_lines(
        self,
        project: SfmProjectBase,
        dict_type: int,
        marker_id: int,
        line_color: List[float] = [0.5, 0.0, 0.1],
        line_to_corners: bool = True,
    ):
        """
        Manually add detection lines from cameras to a specific marker.

        Args:
            project: SfM project containing the marker
            dict_type: ArUco dictionary type
            marker_id: ArUco marker ID
            line_color: RGB color for the lines
            line_to_corners: If True, lines to corners; if False, lines to center
        """
        try:
            marker = project.markers[dict_type][marker_id]
            corners_3d = marker.corners_3d

            if marker is None:
                print(f"Warning: Marker dict={dict_type}, id={marker_id} not found")
                return

            config = {
                "detection_line_color": line_color,
                "line_to_corners": line_to_corners,
            }

            self._add_detection_lines(project, marker, corners_3d, config)
            self.__vis.poll_events()
            self.__vis.update_renderer()

        except Exception as e:
            print(
                f"Error adding detection lines for marker dict={dict_type}, id={marker_id}: {e}"
            )

    def show(self):
        """Display the visualization."""
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()


def draw_camera_viewport(
    extrinsics,
    intrinsics,
    image=None,
    scale=1.0,
    image_width=None,
    image_height=None,
    color=[0.8, 0.2, 0.8],
):
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

        # Create image plane with image as texture (double-sided)
        plane = o3d.geometry.TriangleMesh()
        plane.vertices = pcd.points

        # Create triangles in both directions for double-sided visibility
        triangles_front = [[0, 1, 3], [1, 2, 3]]  # Front-facing
        triangles_back = [[0, 3, 1], [1, 3, 2]]  # Back-facing (reversed winding)
        all_triangles = triangles_front + triangles_back

        plane.triangles = o3d.utility.Vector3iVector(np.asarray(all_triangles))
        plane.compute_vertex_normals()

        # UV coordinates for front-facing triangles
        v_uv_front = np.asarray([[1, 1], [1, 0], [0, 1], [1, 0], [0, 0], [0, 1]])
        # UV coordinates for back-facing triangles (same pattern)
        v_uv_back = np.asarray([[1, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 0]])
        v_uv = np.vstack([v_uv_front, v_uv_back])

        plane.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
        plane.triangle_material_ids = o3d.utility.IntVector(
            [0] * 4
        )  # 4 triangles total
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
