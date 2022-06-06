#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Code for COLMAP readout borrowed from https://github.com/uzh-rpg/colmap_utils/tree/97603b0d352df4e0da87e3ce822a9704ac437933
'''

# Built-in/Generic Imports
import struct
import collections

# Libs
import numpy as np
import open3d as o3d

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image",
                                   ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids",
                                    "point3DiD_to_kpidx"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

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
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def generate_colmap_sparse_pc(points3D):
    sparse_pc = np.zeros((points3D.__len__(), 3))
    sparse_pc_color = np.zeros((points3D.__len__(), 3))

    for idx, pc_idx in enumerate(points3D.__iter__()):
        sparse_pc[idx] = points3D[pc_idx].xyz
        sparse_pc_color[idx] = points3D[pc_idx].rgb / 255.

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(sparse_pc)
    pc.colors = o3d.utility.Vector3dVector(sparse_pc_color)

    return pc


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    # Struct unpack return tuple (https://docs.python.org/3/library/struct.html)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    """
    Original C++ Code can be found here: https://github.com/colmap/colmap/blob/dev/src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        # First 8 bits contain information about the quantity of different camera models
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            # Afterwards the 64 bits contain information about a specific camera
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            # The next  NUM_PARAMS * 8 bits contain information about the camera parameters
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                     format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_images_binary(path_to_model_file):
    """
    Original C++ Code can be found here: https://github.com/colmap/colmap/blob/dev/src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        # First 8 bits contain information about the quantity of different registrated camera models
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            # Image properties: (image_id, qvec, tvec, camera_id)
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            # Normalized rotation quaternion - 4 entries
            qvec = np.array(binary_image_properties[1:5])
            # Translational Part  - 3 entries
            tvec = np.array(binary_image_properties[5:8])
            # Camera ID
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            # Number of 2D image features detected
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            # 2D location of features in image + Feature ID (x,y,id)
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                       format_char_sequence="ddq" * num_points2D)
            # 2D location of features in image
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            # Feature ID
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))

            pt3did_to_kpidx = {}
            for kpidx, ptid in enumerate(point3D_ids.ravel().tolist()):
                if ptid != -1:
                    if ptid in pt3did_to_kpidx:
                        # print("3D point {} already exits in {}, skip".format(
                        # ptid, image_name))
                        continue
                    pt3did_to_kpidx[ptid] = kpidx

            images[image_id] = Image(id=image_id,
                                     qvec=qvec,
                                     tvec=tvec,
                                     camera_id=camera_id,
                                     name=image_name,
                                     xys=xys,
                                     point3D_ids=point3D_ids,
                                     point3DiD_to_kpidx=pt3did_to_kpidx)
    return images


def read_points3d_binary(path_to_model_file):
    """
    Original C++ Code can be found here: https://github.com/colmap/colmap/blob/dev/src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        # Number of points in sparse model
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            # Point ID
            point3D_id = binary_point_line_properties[0]
            # XYZ
            xyz = np.array(binary_point_line_properties[1:4])
            # RGB
            rgb = np.array(binary_point_line_properties[4:7])
            # What kind of error?
            error = np.array(binary_point_line_properties[7])
            # Number of features that observed this 3D Point
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            # Feature ID connected to this point
            track_elems = read_next_bytes(
                fid, num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length)
            # Image ID connected to this 3D Point
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def convert_colmap_extrinsics(frame):
    Rwc = frame.Rwc()
    twc = frame.twc()

    M = np.eye(4)
    M[:3, :3] = Rwc
    M[:3, 3] = twc

    return Rwc, twc, M


def convert_colmap_extrinsics_scaled(frame, scale_factor):
    Rwc = frame.Rwc()
    twc = frame.twc() * scale_factor

    M = np.eye(4)
    M[:3, :3] = Rwc
    M[:3, 3] = twc

    return Rwc, twc, M

# ToDo: Image reader

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

    def qtvec(self):
        return self.qvec.ravel().tolist() + self.tvec.ravel().tolist()

    def Rwc(self):
        return self.qvec2rotmat().transpose()

    def twc(self):
        return np.dot(-self.qvec2rotmat().transpose(), self.tvec)

    def Rcw(self):
        return self.qvec2rotmat()

    def tcw(self):
        return self.tvec

    def Twc(self):
        Twc = np.eye(4)
        Twc[0:3, 3] = self.twc()
        Twc[0:3, 0:3] = self.Rwc()

        return Twc

    def Tcw(self):
        Tcw = np.eye(4)
        Tcw[0:3, 3] = self.tcw()
        Tcw[0:3, 0:3] = self.Rcw()

        return Tcw


class Intrinsics:
    def __init__(self, camera):
        self._cx = None
        self._cy = None
        self._fx = None
        self._fy = None

        self.camera = camera
        self.load_from_colmap(camera=self.camera)

    def load_from_colmap(self, camera):
        self.fx = camera.params[0]
        self.fy = camera.params[1]
        self.cx = camera.params[2]
        self.cy = camera.params[3]

    @property
    def cx(self):
        return self._cx

    @cx.setter
    def cx(self, cx):
        self._cx = cx

    @property
    def cy(self):
        return self._cy

    @cy.setter
    def cy(self, cy):
        self._cy = cy

    @property
    def fx(self):
        return self._fx

    @fx.setter
    def fx(self, fx):
        self._fx = fx

    @property
    def fy(self):
        return self._fy

    @fy.setter
    def fy(self, fy):
        self._fy = fy

    @property
    def K(self):
        K = np.asarray([[self.fx, 0, self.cx],
                        [0, self.fy, self.cy],
                        [0, 0, 1]])

        return K


def read_reconstruction_data(base_path):
    '''
        Path to colmap folder. It must contain folder 'sparse' with files 'cameras.bin', 'images.bin', 'points3D.bin'.

    Returns cameras, images, points3D

    :param path:
    :return: cameras, images, points3D
    '''
    reco_base_path = path.Path(base_path)
    sparse_base_path = reco_base_path.joinpath('sparse')
    camera_path = sparse_base_path.joinpath('cameras.bin')
    image_path = sparse_base_path.joinpath('images.bin')
    points3D_path = sparse_base_path.joinpath('points3D.bin')

    points3D = read_points3d_binary(points3D_path)
    cameras = read_cameras_binary(camera_path)
    images = read_images_binary(image_path)

    return cameras, images, points3D


if __name__ == '__main__':
    reco_base_path = path.Path('../../data/')
    sparse_base_path = reco_base_path.joinpath('sparse')
    camera_path = sparse_base_path.joinpath('cameras.bin')
    image_path = sparse_base_path.joinpath('images.bin')
    points3D_path = sparse_base_path.joinpath('points3D.bin')

    points3D = read_points3d_binary(points3D_path)
    cameras = read_cameras_binary(camera_path)
    images = read_images_binary(image_path)
