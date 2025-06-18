import collections
from abc import ABC, abstractmethod
from typing import Dict, Tuple
import logging
# from ..aruco_localizer import ArucoLocalizer
import numpy as np
import cv2
from ..utils import qvec2rotmat, rotmat2qvec
from ..aruco import localize_aruco_markers

# Standardized data structures - all SfM software must produce these
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
BaseCamera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)
Marker = collections.namedtuple(
    "Marker", [
        "id",           # ArUco marker ID
        "xyz",          # Center position in 3D
        "corners_3d",   # 4x3 array of corner positions in 3D
        "dict_type",    # ArUco dictionary type
        "image_ids",    # List of image IDs where detected
        "point2D_idxs"  # List of 2D corner coordinates for each detection
    ]
)

class Camera(BaseCamera):
    """Standardized camera class - all SfM software must produce this format."""
    
    @property
    def K(self):
        """Get intrinsics matrix K."""
        if self.model == "PINHOLE":
            fx, fy, cx, cy = self.params
            return np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        elif self.model == "SIMPLE_PINHOLE":
            f, cx, cy = self.params
            return np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1]
            ])
        elif self.model == "SIMPLE_RADIAL":
            f, cx, cy, k1 = self.params  # k1 is distortion, not used in K matrix
            return np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1]
            ])
    
        else:
            raise NotImplementedError(f"Camera model {self.model} not implemented")
    
    @property
    def intrinsics(self):
        """Get intrinsics object with K matrix."""
        class Intrinsics:
            def __init__(self, K):
                self.K = K
        return Intrinsics(self.K)


class Image(BaseImage):
    """Standardized image class - all SfM software must produce this format."""
    @property
    def world_extrinsics(self):
        """Get 4x4 camera to world transformation matrix."""
        R = qvec2rotmat(self.qvec)
        t = self.tvec.reshape(3, 1)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return np.linalg.inv(T)
    
    def get_camera_center(self):
        """Get camera center in world coordinates."""
        R = qvec2rotmat(self.qvec)
        return -R.T @ self.tvec

# Camera models
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}

CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)

class SfmProjectBase(ABC):
    """
    Abstract base class for Structure from Motion projects.
    
    All SfM software implementations (COLMAP, OpenMVG, etc.) must inherit from this
    and convert their data to the standardized Camera, Image, Point3D format.
    """
    
    def detect_markers(self, 
                      dict_type: int = cv2.aruco.DICT_4X4_50,
                      detector_params: cv2.aruco.DetectorParameters = None,
                      progress_bar: bool = True,
                      num_processes: int = None,
                      use_multiprocessing: bool = True) -> Dict[int, Tuple[float, np.ndarray]]:
        """
        Detect ArUco markers in the project images for a single dictionary type.
        
        :param dict_type: ArUco dictionary type to use
        :param detector: Pre-configured ArucoDetector (if None, will be created)
        :param detector_params: Detector parameters (if detector not provided)
        :param progress_bar: Show progress bars
        :param num_processes: Number of processes for multiprocessing (None = auto)
        :param use_multiprocessing: Whether to use multiprocessing (can disable if issues occur)
        :return: Dictionary mapping aruco_id -> (distance, corners_3d)
        """
        
        # Create detector if not provided
        if detector_params is None:
            detector_params = cv2.aruco.DetectorParameters()
            
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        
        # Disable multiprocessing if requested
        if not use_multiprocessing:
            num_processes = 1
        
        # Run ArUco detection and localization
        raw_results = localize_aruco_markers(
            project=self,
            dict_type=dict_type,
            detector=detector,
            progress_bar=progress_bar,
            num_processes=num_processes
        )
        
        # Store results in project using Marker namedtuples
        self._store_marker_results(dict_type, raw_results)
        
        # Return simplified results for backward compatibility
        results = {}
        for aruco_id, data in raw_results.items():
            results[aruco_id] = data['corners_3d']
        
        logging.info(f"ArUco detection complete. Found {len(results)} markers for dict type {dict_type}.")
        return results
    
    def _store_marker_results(self, dict_type: int, raw_results: Dict):
        """
        Store marker results using Marker namedtuples with 3D corners included.
        
        :param dict_type: ArUco dictionary type
        :param raw_results: Raw results from localize_aruco_markers function
        """
        # Initialize storage if needed
        if not hasattr(self, '_markers'):
            self._markers = {}
            
        # Initialize for this dictionary type
        if dict_type not in self._markers:
            self._markers[dict_type] = {}
        
        for aruco_id, data in raw_results.items():
            # Create Marker namedtuple with 3D corners included
            marker = Marker(
                id=aruco_id,
                xyz=data['center_xyz'],
                corners_3d=data['corners_3d'],
                dict_type=dict_type,
                image_ids=data['image_ids'],
                point2D_idxs=data['corner_pixels']
            )
            
            self._markers[dict_type][aruco_id] = marker
            
            logging.info(f"Stored marker dict={dict_type}, id={aruco_id}")
    
    def get_markers(self, dict_type: int = None, aruco_id: int = None):
        """
        Get stored markers from the project.
        
        :param dict_type: Optional dictionary type filter
        :param aruco_id: Optional specific ArUco ID filter
        :return: Marker namedtuple(s)
        """
        if not hasattr(self, '_markers') or not self._markers:
            return {} if dict_type is None else (None if aruco_id is not None else {})
            
        if dict_type is not None:
            dict_markers = self._markers.get(dict_type, {})
            if aruco_id is not None:
                return dict_markers.get(aruco_id, None)
            return dict_markers
        else:
            return self._markers
    
    def get_marker_corners_3d(self, dict_type: int, aruco_id: int) -> np.ndarray:
        """
        Get 3D corners for a specific marker.
        
        :param dict_type: ArUco dictionary type
        :param aruco_id: ArUco marker ID
        :return: 4x3 array of corner positions
        """
        marker = self.get_markers(dict_type, aruco_id)
        if marker is None:
            raise ValueError(f"Marker dict={dict_type}, id={aruco_id} not found")
        return marker.corners_3d
    
    def get_marker_distance(self, dict_type: int, aruco_id: int) -> float:
        """
        Get average edge distance for a specific marker.
        
        :param dict_type: ArUco dictionary type
        :param aruco_id: ArUco marker ID
        :return: Average distance between adjacent corners
        """
        from .aruco_detection import calculate_marker_distance
        corners_3d = self.get_marker_corners_3d(dict_type, aruco_id)
        return calculate_marker_distance(corners_3d)
    
    def __init__(self, project_path):
        self._project_path = project_path
        self._cameras = {}
        self._images = {}
        self._points3D = {}
        self._markers = {}  # Structure: {dict_type: {aruco_id: Marker}}
        self._load_data()
        self._verify_data_loaded()
        
    def clear_markers(self, dict_type: int = None):
        """
        Clear stored markers.
        
        :param dict_type: Optional specific dictionary type to clear (if None, clears all)
        """
        if not hasattr(self, '_markers'):
            return
            
        if dict_type is not None:
            if dict_type in self._markers:
                del self._markers[dict_type]
            logging.info(f"Cleared markers for dict type {dict_type}")
        else:
            self._markers = {}
            logging.info("Cleared all markers")

    @abstractmethod
    def load_image_by_id(self, image_id):
        # return opencv image
        pass
        
    @abstractmethod
    def _load_data(self):
        """Load project data from files."""
        pass
    
    def _verify_data_loaded(self):
        """Check that data was successfully loaded."""
        if not self._cameras and not self._images and not self._points3D:
            raise ValueError(f"No data loaded from {self._project_path}")
        print(f"Loaded {len(self._cameras)} cameras, {len(self._images)} images, {len(self._points3D)} points")
    
    @abstractmethod
    def save(self):
        """Save project data to files."""
        pass
    @abstractmethod
    def _extra_transforms(self,transform_matrix):
        pass
    def transform(self, transform_matrix):
        """Apply 4x4 transformation matrix to poses, 3D points, and markers."""
        if transform_matrix.shape != (4, 4):
            raise ValueError("Transform matrix must be 4x4")
        
        # Transform markers
        if hasattr(self, '_markers'):
            for dict_type in self._markers:
                for aruco_id in self._markers[dict_type]:
                    marker = self._markers[dict_type][aruco_id]
                    
                    # Transform 3D corners
                    transformed_corners = []
                    for corner in marker.corners_3d:
                        corner_h = np.append(corner, 1)
                        transformed_h = transform_matrix @ corner_h
                        transformed_corners.append(transformed_h[:3])
                    
                    transformed_corners_3d = np.array(transformed_corners)
                    
                    # Transform center position
                    center_h = np.append(marker.xyz, 1)
                    transformed_center_h = transform_matrix @ center_h
                    new_center = transformed_center_h[:3]
                    
                    # Create new marker with transformed data
                    self._markers[dict_type][aruco_id] = marker._replace(
                        xyz=new_center,
                        corners_3d=transformed_corners_3d
                    )


        
        # Transform camera poses
        for img_id, img in self._images.items():
            # Apply transformation: new_c2w = transform * old_c2w
            new_c2w = transform_matrix @ img.world_extrinsics
            
            # Convert back to world-to-camera for qvec/tvec storage
            new_w2c = np.linalg.inv(new_c2w)
            new_R_w2c = new_w2c[:3, :3]
            new_t_w2c = new_w2c[:3, 3]
            
            # Update image with new qvec/tvec
            new_qvec = rotmat2qvec(new_R_w2c)
            self._images[img_id] = img._replace(qvec=new_qvec, tvec=new_t_w2c)
        
        # Transform 3D points
        for pt_id, pt in self._points3D.items():
            point_h = np.append(pt.xyz, 1)
            transformed_h = transform_matrix @ point_h
            new_xyz = transformed_h[:3]
            self._points3D[pt_id] = pt._replace(xyz=new_xyz)
        self._extra_transforms(transform_matrix)
        logging.info("Applied transformation to poses, 3D points, and markers")

    @property
    def cameras(self):
        return self._cameras
    
    @property
    def images(self):
        return self._images
    
    @property
    def points3D(self):
        return self._points3D
    
    @property
    def markers(self):
        """Get all markers as Marker namedtuples."""
        if not hasattr(self, '_markers'):
            return {}
        return self._markers