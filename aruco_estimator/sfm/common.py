import collections
from abc import ABC, abstractmethod

import numpy as np

from ..utils import qvec2rotmat, rotmat2qvec

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
    def detect_markers(self,dict, detector):
        pass
    def __init__(self, project_path):
        self._project_path = project_path
        self._cameras = {}
        self._images = {}
        self._points3D = {}
        # self.images_path = None
        self._load_data()
        self._verify_data_loaded()
        
        # self._detected_tags = dict()
    @abstractmethod
    def load_image_by_id(self,image_id):
        # return opencv image
        pass
    @abstractmethod
    def _load_data(self):
        """Load project data from files."""
        pass
    
    def _verify_data_loaded(self):
        """Check that data was successfully loaded."""
        # assert self.images_path is not None
        if not self._cameras and not self._images and not self._points3D:
            raise ValueError(f"No data loaded from {self._project_path}")
        print(f"Loaded {len(self._cameras)} cameras, {len(self._images)} images, {len(self._points3D)} points")
    
    @abstractmethod
    def save(self):
        """Save project data to files."""
        pass
    

    def transform(self, transform_matrix):
        """Apply 4x4 transformation matrix to poses and 3D points."""
        if transform_matrix.shape != (4, 4):
            raise ValueError("Transform matrix must be 4x4")
        
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
        

   
    @property
    def cameras(self):
        return self._cameras
    
    @property
    def images(self):
        return self._images
    
    @property
    def points3D(self):
        return self._points3D
    