#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""
# Built-in/Generic Imports
from copy import deepcopy
import os
import time
from functools import partial
from functools import wraps
from multiprocessing import Pool

# Libs
from tqdm import tqdm
# Own modules
from colmap_wrapper.colmap.colmap import COLMAPProject
from colmap_wrapper.colmap.utils import generate_colmap_sparse_pc

from aruco_estimator.aruco import *
from aruco_estimator.opt import *
from aruco_estimator.base import *

DEBUG = False


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        if DEBUG:
            print(f'Function {func.__name__}{args} {kwargs} took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


class ArucoScaleFactor(ScaleFactorBase):
    def __init__(self, photogrammetry_software: COLMAPProject, aruco_size: float, dense_path: str = 'fused.ply'):
        """
        This class is used to determine 3D points of the aruco marker, which are used to compute a scaling factor.
        In the following the workflow is shortly described.

                  ---------------------             Load data from COLMAp project. They include extrinsic, intrinsic
                  |  Load COLMAP Data |             parameters, images, sparse and dense point cloud
                  ---------------------
                            |
                            v
                --------------------------
                | Aruco Marker Detection |
                --------------------------
                            |
                            v
                    ---------------
                    | Ray casting |
                    ---------------
                            |
                            v
                -------------------------------
                | LS for Interection of Lines |
                -------------------------------

        :param photogrammetry_software:
        :param aruco_size:
        :param dense_path:
        """
        # COLMAP.__init__(self, project_path=project_path, dense_pc=dense_path)
        super().__init__(photogrammetry_software)

        self.aruco_marker_detected = None

        # Values to calculate 3D point of intersection
        self.images_scaled = None
        self.P0 = np.array([])
        self.N = np.array([])

        # Aruco specific
        self.aruco_distance = None
        self.aruco_corners_3d = None
        self.aruco_dict_type = aruco.DICT_4X4_1000

        # Results
        self.scale_factor = None
        self.dense_scaled = None
        self.sparse_scaled = None

        # Multi Processing
        self.progress_bar = True
        self.num_processes = 8 if os.cpu_count() > 8 else os.cpu_count()
        print('Num process: ', self.num_processes)
        self.image_names = []
        # Prepare parsed data for multi processing
        for image_idx in self.photogrammetry_software.images.keys():
            self.image_names.append(self.photogrammetry_software._src_image_path.joinpath(
                self.photogrammetry_software.images[image_idx].name).__str__())

        #if os.path.exists(self.photogrammetry_software._project_path.joinpath('aruco_size.txt')):
        #    self.aruco_size = float(
        #        open(self.photogrammetry_software._project_path.joinpath('aruco_size.txt'), 'r').read())
        #else:
        self.aruco_size = aruco_size

    def run(self) -> [np.ndarray, None]:
        """
        Starts the aruco extraction, ray casting and intersection of lines.

        :return:
        """
        self.__detect()

        # if not self.aruco_marker_detected:
        #    return self.aruco_marker_detected

        self.__ray_cast()
        self.aruco_corners_3d = intersect_parallelized(P0=self.P0.reshape(len(self.P0) // 3, 3),
                                                       N=self.N.reshape(len(self.N) // 12, 4, 3))
        self.aruco_distance = self.__evaluate(self.aruco_corners_3d)

        return self.aruco_distance

    @timeit
    def __detect(self):
        """
        Detects the aruco corners in the image and extracts the aruco id. Aftwards the image, id and tuple of corners
        are saved into the Image class to parse the data through the algorithm. If no aruco marker is detected it
        returns None.

        :return:
        """
        with Pool(self.num_processes) as p:
            result = list(tqdm(p.imap(
                partial(detect_aruco_marker, dict_type=self.aruco_dict_type),
                self.image_names), total=len(self.image_names), disable=not self.progress_bar))

        if len(result) != len(self.photogrammetry_software.images):
            raise ValueError("Thread return has not the same length as the input parameters!")

        # Checks if all tuples in list are None
        # if not all(all(v) for v in result):
        #    self.aruco_marker_detected = False
        # else:
        #    self.aruco_marker_detected = True

        for image_idx in self.photogrammetry_software.images.keys():
            self.photogrammetry_software.images[image_idx].aruco_corners = result[image_idx - 1][0]
            self.photogrammetry_software.images[image_idx].aruco_id = result[image_idx - 1][1]
            self.photogrammetry_software.images[image_idx].image_path = self.image_names[image_idx - 1]
            # self.images[image_idx].image = cv2.resize(result[image_idx - 1][2], (0, 0), fx=0.3, fy=0.3)

    def __ray_cast(self):
        """
        This function casts a ray from the origin of the camera center C_i (also the translational part of the extrinsic
        matrix) through the detected aruco corners in the image coordinates x = (x,y,1) which direction is defined
        trough n = x @ K^-1 @ R.T. K^-1 is the inverse of the intrinsic matrix and R is the extrinsic matrix
        (only rotation) of the current frame. Afterwards the origin C_i and the normalized direction vector for all
        aruco corners are saved.

        :return:
        """
        for image_idx in self.photogrammetry_software.images.keys():
            if self.photogrammetry_software.images[image_idx].aruco_corners is not None:
                p0, n = ray_cast_aruco_corners(extrinsics=self.photogrammetry_software.images[image_idx].extrinsics,
                                               intrinsics=self.photogrammetry_software.images[image_idx].intrinsics.K,
                                               corners=self.photogrammetry_software.images[image_idx].aruco_corners)
                self.photogrammetry_software.images[image_idx].p0 = p0
                self.photogrammetry_software.images[image_idx].n = n

                self.P0 = np.append(self.P0, p0)
                self.N = np.append(self.N, n)

    @staticmethod
    def __evaluate(aruco_corners_3d: np.ndarray) -> np.ndarray:
        """
        Calculates the L2 norm between every neighbouring aruco corner. Finally the distances are averaged and returned

        :return:
        """
        dist1 = np.linalg.norm(aruco_corners_3d[0] - aruco_corners_3d[1])
        dist2 = np.linalg.norm(aruco_corners_3d[1] - aruco_corners_3d[2])
        dist3 = np.linalg.norm(aruco_corners_3d[2] - aruco_corners_3d[3])
        dist4 = np.linalg.norm(aruco_corners_3d[3] - aruco_corners_3d[0])

        # Average
        return np.mean([dist1, dist2, dist3, dist4])

    def analyze(self):
        """

        @param true_scale: true scale of aruco marker in centimeter!
        @return:
        """

        sf = np.zeros(len(self.P0) // 3)

        for i in range(2, len(self.P0) // 3 + 1):
            P0_i = self.P0.reshape(len(self.P0) // 3, 3)[:i]
            N_i = self.N.reshape(len(self.N) // 12, 4, 3)[:i]
            aruco_corners_3d = intersect_parallelized(P0=P0_i, N=N_i)
            aruco_dist = self.__evaluate(aruco_corners_3d)
            sf[i - 1] = (self.aruco_size) / aruco_dist

        plt.figure()
        plt.title('Scale Factor Estimation over Number of Images')
        plt.xlabel('# Images')
        plt.ylabel('Scale Factor')
        plt.grid()
        plt.plot(np.linspace(1, len(self.P0) // 3, len(self.P0) // 3)[1:], sf[1:])
        plt.show()

    def get_dense_scaled(self):
        return self.dense_scaled

    def get_sparse_scaled(self):
        return generate_colmap_sparse_pc(self.sparse_scaled)

    def apply(self) -> Tuple[o3d.pybind.geometry.PointCloud, float]:
        """
        This function can be used if the scaling of the dense point cloud should be applied directly + the extrinsic
        paramters should be scaled.

        ToDo: save them to a folder!
        @param true_scale:
        @return:
        """

        self.scale_factor = (self.aruco_size) / self.aruco_distance
        self.photogrammetry_software.dense_scaled = deepcopy(self.photogrammetry_software.dense)
        self.photogrammetry_software.dense_scaled.scale(scale=self.scale_factor, center=np.asarray([0., 0., 0.]))

        # self.sparse_scaled = deepcopy(self.get_sparse())
        # self.sparse_scaled.scale(scale=self.scale_factor, center=np.asarray([0., 0., 0.]))

        self.sparse_scaled = deepcopy(self.photogrammetry_software.sparse)
        for num in self.photogrammetry_software.sparse.keys():
            self.sparse_scaled[num].xyz = self.sparse_scaled[num].xyz * self.scale_factor

        # self.sparse_scaled.scale(scale=self.scale_factor, center=np.asarray([0., 0., 0.]))

        # ToDo: Scale tvec and save
        self.photogrammetry_software.images_scaled = deepcopy(self.photogrammetry_software.images)
        for idx in self.photogrammetry_software.images_scaled.keys():
            self.photogrammetry_software.images_scaled[idx].tvec = self.photogrammetry_software.images[
                                                                       idx].tvec * self.scale_factor

        return self.photogrammetry_software.dense_scaled, self.scale_factor

    def write_data(self):

        pcd_scaled = self.photogrammetry_software._project_path
        cameras_scaled = self.photogrammetry_software._project_path.joinpath('sparse_scaled/cameras')
        images_scaled = self.photogrammetry_software._project_path.joinpath('sparse_scaled/images')
        points_scaled = self.photogrammetry_software._project_path.joinpath('sparse_scaled/points3D')

        cameras_scaled.mkdir(parents=True, exist_ok=True)
        images_scaled.mkdir(parents=False, exist_ok=True)
        points_scaled.mkdir(parents=False, exist_ok=True)

        for image_idx in self.photogrammetry_software.images_scaled.keys():
            filename = images_scaled.joinpath('image_{:04d}.txt'.format(image_idx - 1))
            np.savetxt(filename, self.photogrammetry_software.images[image_idx].extrinsics.flatten())

        o3d.io.write_point_cloud(os.path.join(pcd_scaled, 'scaled.ply'), self.photogrammetry_software.dense_scaled)

        # Save scale factor
        scale_factor_file_name = self.photogrammetry_software._project_path.joinpath('sparse_scaled/scale_factor.txt')
        np.savetxt(scale_factor_file_name, np.array([self.scale_factor]))


if __name__ == '__main__':
    from colmap_wrapper.colmap import COLMAPProject
    from aruco_estimator.visualization import ArucoVisualization

    project = COLMAPProject(project_path='../data/door', image_resize=0.4)

    aruco_scale_factor = ArucoScaleFactor(photogrammetry_software=project, aruco_size=0.15)
    aruco_distance = aruco_scale_factor.run()
    print('Mean distance between aruco markers: ', aruco_distance)

    aruco_scale_factor.analyze()

    dense, scale_factor = aruco_scale_factor.apply()
    print('Point cloud and poses are scaled by: ', scale_factor)

    vis = ArucoVisualization(aruco_colmap=aruco_scale_factor)
    vis.visualization(frustum_scale=0.7, point_size=0.1)

    aruco_scale_factor.write_data()
