import argparse
import collections
import os
import struct
import numpy as np
from abc import ABC, abstractmethod
from ..utils import qvec2rotmat, rotmat2qvec 

import numpy as np
from abc import ABC, abstractmethod


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
    def get_camera_matrix(self):
        """Get 4x4 camera transformation matrix."""
        R = qvec2rotmat(self.qvec)
        t = self.tvec.reshape(3, 1)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T
    
    def get_camera_center(self):
        """Get camera center in world coordinates."""
        R = qvec2rotmat(self.qvec)
        return -R.T @ self.tvec
    
    @property
    def extrinsics(self):
        """Get 4x4 extrinsics matrix."""
        return self.get_camera_matrix()


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
    
    def __init__(self, project_path):
        self._project_path = project_path
        self._cameras = {}
        self._images = {}
        self._points3D = {}
        self.scale_factor = 1.0
        self._load_data()
        self._verify_data_loaded()
    
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
    
    def transform(self, transform_matrix):
        return
        """Apply 4x4 transformation matrix to poses and 3D points."""
        if transform_matrix.shape != (4, 4):
            raise ValueError("Transform matrix must be 4x4")
        
        # Extract the scale factor from the transformation matrix
        scale_factor = np.linalg.norm(transform_matrix[:3, 0])
        
        # Prepare a rotation-only transform (scale removed)
        normalized_transform = np.eye(4)
        normalized_transform[:3, :3] = transform_matrix[:3, :3] / scale_factor
        normalized_transform[:3, 3] = transform_matrix[:3, 3] / scale_factor
        
        # Compute the inverse transform for camera poses
        inverse_normalized_transform = np.linalg.inv(normalized_transform)
        
        # Transform camera poses
        for img_id, img in self._images.items():
            # Original pose
            R = qvec2rotmat(img.qvec)
            t = img.tvec
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            
            # Apply inverse of normalization transform
            rotated_pose = pose @ inverse_normalized_transform
            rotated_pose[:3, 3] *= scale_factor  # Reapply scale only to translation
            
            # Extract new rotation and translation
            new_R = rotated_pose[:3, :3]
            new_t = rotated_pose[:3, 3]
            
            # Create new image with transformed pose
            # new_img = img._replace(qvec=rotmat2qvec(new_R), tvec=new_t)
            # self._images[img_id] = Image(*new_img)
            self._images[img_id] = img._replace(qvec=rotmat2qvec(new_R), tvec=new_t)
        
        # Transform 3D points
        for pt_id, pt in self._points3D.items():
            point_h = np.append(pt.xyz, 1)
            transformed_h = normalized_transform @ point_h
            new_xyz = transformed_h[:3]
            new_pt = pt._replace(xyz=new_xyz)
            self._points3D[pt_id] = new_pt
        
        # Update scale factor
        self.scale_factor *= scale_factor
    
    @property
    def cameras(self):
        return self._cameras
    
    @property
    def images(self):
        return self._images
    
    @property
    def points3D(self):
        return self._points3D
    

# def normalize_poses_and_points(cameras, images, points3D, transform: np.ndarray):
#     """Apply normalization transform to camera poses and 3D points."""

#     # Extract the scale factor from the transformation matrix
#     scale_factor = np.linalg.norm(transform[:3, 0])
#     logging.info(f"Extracted scale factor from transform: {scale_factor:.4f}")

#     # Prepare a rotation-only transform (scale removed)
#     normalized_transform = np.eye(4)
#     normalized_transform[:3, :3] = transform[:3, :3] / scale_factor
#     normalized_transform[:3, 3] = transform[:3, 3] / scale_factor

#     # Compute the inverse transform for camera poses
#     inverse_normalized_transform = np.linalg.inv(normalized_transform)

#     # Transform camera poses
#     transformed_images = {}
#     for image_id, image in images.items():
#         # Original pose
#         R = qvec2rotmat(image.qvec)
#         t = image.tvec
#         pose = np.eye(4)
#         pose[:3, :3] = R
#         pose[:3, 3] = t

#         # Apply inverse of normalization transform
#         rotated_pose = pose @ inverse_normalized_transform
#         rotated_pose[:3, 3] *= scale_factor  # Reapply scale only to translation

#         # Extract new rotation and translation
#         new_R = rotated_pose[:3, :3]
#         new_t = rotated_pose[:3, 3]

#         # Create new image with transformed pose
#         transformed_images[image_id] = Image(
#             id=image.id,
#             qvec=rotmat2qvec(new_R),
#             tvec=new_t,
#             camera_id=image.camera_id,
#             name=image.name,
#             xys=image.xys,
#             point3D_ids=image.point3D_ids,
#         )

#     # Transform 3D points
#     transformed_points3D = {}
#     for point3D_id, point3D in points3D.items():
#         point_h = np.append(point3D.xyz, 1)
#         transformed_h = normalized_transform @ point_h
#         new_xyz = transformed_h[:3]
#         transformed_points3D[point3D_id] = Point3D(
#             id=point3D.id,
#             xyz=new_xyz,
#             rgb=point3D.rgb,
#             error=point3D.error,
#             image_ids=point3D.image_ids,
#             point2D_idxs=point3D.point2D_idxs,
#         )

#     return cameras, transformed_images, transformed_points3D

