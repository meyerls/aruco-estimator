#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
from functools import wraps
import time
import os

# Libs
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

# Own modules
try:
    from colmap import *
    from helper.utils import *
    from helper.visualization import *
    from helper.aruco import *
    from helper.opt import *
    from base import *
except ImportError:
    from .colmap import *
    from .helper.utils import *
    from .helper.visualization import *
    from .helper.aruco import *
    from .helper.opt import *
    from .base import *

DEBUG = False


class ScaleFactorBase:
    def __init__(self):
        """
        Base class for scale factor estimation.

            ---------------
            |    Detect   |
            ---------------
                    |
                    v
            ---------------
            |     Run     |
            ---------------
                    |
                    v
            ---------------
            |   Evaluate  |
            ---------------
                    |
                    v
            ---------------
            |     Apply   |
            ---------------
        """
        pass

    def detect(self):
        return NotImplemented

    def run(self):
        return NotImplemented

    def evaluate(self):
        return NotImplemented

    def apply(self, *args, **kwargs):
        return NotImplemented


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


class ArucoScaleFactor(ScaleFactorBase, COLMAP):
    def __init__(self, project_path: str, dense_pc: str = 'fused.ply'):
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

        :param project_path:
        :param dense_pc:
        """
        COLMAP.__init__(self, project_path=project_path, dense_pc=dense_pc)

        # Values to calculate 3D point of intersection
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
        self.num_processes = os.cpu_count()
        self.image_names = []
        # Prepare parsed data for multi processing
        for image_idx in self.images.keys():
            self.image_names.append(self._COLMAP__src_image_path.joinpath(self.images[image_idx].name).__str__())

    def __ray_cast(self):
        """
        This function casts a ray from the origin of the camera center C_i (also the translational part of the extrinsic
        matrix) through the detected aruco corners in the image coordinates x = (x,y,1) which direction is defined
        trough n = x @ K^-1 @ R.T. K^-1 is the inverse of the intrinsic matrix and R is the extrinsic matrix
        (only rotation) of the current frame. Afterwards the origin C_i and the normalized direction vector for all
        aruco corners are saved.

        :return:
        """
        for image_idx in self.images.keys():
            camera_intrinsics = Intrinsics(camera=self.cameras[self.images[image_idx].camera_id])

            Rwc, twc, M = convert_colmap_extrinsics(frame=self.images[image_idx])

            if self.images[image_idx].aruco_corners != None:
                p0, n = ray_cast_aruco_corners(extrinsics=M,
                                               intrinsics=camera_intrinsics.K,
                                               corners=self.images[image_idx].aruco_corners)

                self.P0 = np.append(self.P0, p0)
                self.N = np.append(self.N, n)

    def __visualization_scaled_scene(self, frustum_scale: float = 0.5):
        """
        This visualization function show the scaled dense and scaled extrinsic parameters.


        :param frustum_scale:
        :return:
        """

        geometries = []
        geometries.append(self.dense_scaled)

        for image_idx in self.images.keys():
            camera_intrinsics = Intrinsics(camera=self.cameras[self.images[image_idx].camera_id])

            Rwc, twc, M = convert_colmap_extrinsics_scaled(frame=self.images[image_idx],
                                                           scale_factor=self.scale_factor)

            line_set, sphere, mesh = draw_camera_viewport(extrinsics=M,
                                                          intrinsics=camera_intrinsics.K,
                                                          image=self.images[image_idx].image,
                                                          scale=frustum_scale)

            # aruco_line_set = ray_cast_aruco_corners_visualization(extrinsics=M,
            #                                                      intrinsics=camera_intrinsics.K,
            #                                                      corners=self.images[image_idx].aruco_corners)

            # geometries.append(aruco_line_set)
            geometries.append(mesh)
            geometries.append(line_set)
            geometries.append(sphere)

        sphere_size = 0.01

        aruco_sphere1 = create_sphere_mesh(t=self.aruco_corners_3d[0] * self.scale_factor,
                                           color=[0, 0, 0],
                                           radius=sphere_size)
        aruco_sphere2 = create_sphere_mesh(t=self.aruco_corners_3d[1] * self.scale_factor,
                                           color=[1, 0, 0],
                                           radius=sphere_size)
        aruco_sphere3 = create_sphere_mesh(t=self.aruco_corners_3d[2] * self.scale_factor,
                                           color=[0, 0, 1],
                                           radius=sphere_size)
        aruco_sphere4 = create_sphere_mesh(t=self.aruco_corners_3d[3] * self.scale_factor,
                                           color=[1, 1, 1],
                                           radius=sphere_size)

        geometries.append(aruco_sphere1)
        geometries.append(aruco_sphere2)
        geometries.append(aruco_sphere3)
        geometries.append(aruco_sphere4)

        aruco_rect = generate_line_set(points=[self.aruco_corners_3d[0] * self.scale_factor,
                                               self.aruco_corners_3d[1] * self.scale_factor,
                                               self.aruco_corners_3d[2] * self.scale_factor,
                                               self.aruco_corners_3d[3] * self.scale_factor],
                                       lines=[[0, 1], [1, 2], [2, 3], [3, 0]], color=[1, 0, 0])

        '''
        # Plot/show text of distance in 3 Reco. Currently not working as text is rotated wrongly. tbd!

        pos_text = (self.aruco_corners_3d[0] + (
                    self.aruco_corners_3d[1] - self.aruco_corners_3d[0]) / 2) * self.scale_factor
        pcd_tree = o3d.geometry.KDTreeFlann(self.dense_scaled)
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pos_text, 100)

        dir_vec = []
        for i in idx:
            dir_vec.append(self.dense_scaled.normals[i])
        dir_vec = np.asarray(dir_vec).mean(axis=0)

        dist = np.linalg.norm(
            self.aruco_corners_3d[0] * self.scale_factor - self.aruco_corners_3d[1] * self.scale_factor)
        pcd_text = text_3d(text='{:.4f} cm'.format(dist*100),
                           pos=pos_text,
                           direction=dir_vec)
        geometries.append(pcd_text)

        '''

        geometries.append(aruco_rect)

        viewer = o3d.visualization.Visualizer()

        viewer.create_window()
        for geometry in geometries:
            viewer.add_geometry(geometry)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        opt.point_size = 3.
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        viewer.run()
        viewer.destroy_window()

    def visualization(self, frustum_scale: float = 1, point_size: float = 1.,sphere_size:float=0.02):
        """

        :param frustum_scale:
        :param point_size:
        :return:
        """

        geometries = []
        geometries.append(self.get_dense())

        for image_idx in self.images.keys():
            camera_intrinsics = Intrinsics(camera=self.cameras[self.images[image_idx].camera_id])

            Rwc, twc, M = convert_colmap_extrinsics(frame=self.images[image_idx])

            line_set, sphere, mesh = draw_camera_viewport(extrinsics=M,
                                                          intrinsics=camera_intrinsics.K,
                                                          image=self.images[image_idx].image,
                                                          scale=frustum_scale)

            aruco_line_set = ray_cast_aruco_corners_visualization(extrinsics=M,
                                                                  intrinsics=camera_intrinsics.K,
                                                                  corners=self.images[image_idx].aruco_corners,
                                                                  corners3d=self.aruco_corners_3d)

            geometries.append(aruco_line_set)
            geometries.append(mesh)
            geometries.append(line_set)
            geometries.append(sphere)

        aruco_sphere1 = create_sphere_mesh(t=self.aruco_corners_3d[0],
                                           color=[0, 0, 0],
                                           radius=sphere_size)
        aruco_sphere2 = create_sphere_mesh(t=self.aruco_corners_3d[1],
                                           color=[1, 0, 0],
                                           radius=sphere_size)
        aruco_sphere3 = create_sphere_mesh(t=self.aruco_corners_3d[2],
                                           color=[0, 0, 1],
                                           radius=sphere_size)
        aruco_sphere4 = create_sphere_mesh(t=self.aruco_corners_3d[3],
                                           color=[1, 1, 1],
                                           radius=sphere_size)

        aruco_rect = generate_line_set(points=[self.aruco_corners_3d[0],
                                               self.aruco_corners_3d[1],
                                               self.aruco_corners_3d[2],
                                               self.aruco_corners_3d[3]],
                                       lines=[[0, 1], [1, 2], [2, 3], [3, 1]],
                                       color=[1, 0, 0])
        geometries.append(aruco_sphere1)
        geometries.append(aruco_sphere2)
        geometries.append(aruco_sphere3)
        geometries.append(aruco_sphere4)
        geometries.append(aruco_rect)

        viewer = o3d.visualization.Visualizer()

        viewer.create_window()
        for geometry in geometries:
            viewer.add_geometry(geometry)
        opt = viewer.get_render_option()
        # opt.show_coordinate_frame = True
        opt.point_size = point_size
        # opt.background_color = np.asarray([0.5, 0.5, 0.5])
        viewer.run()
        viewer.destroy_window()

    @timeit
    def detect(self):
        """
        Detects the aruco corners in the image and extracts the aruco id. Aftwards the image, id and tuple of corners
        are saved into the Image class to parse the data through the algorithm. If no aruco marker is detected it
        returns None.

        :return:
        """
        from functools import partial

        with Pool(self.num_processes) as p:
            result = p.map(partial(detect_aruco_marker, dict_type=self.aruco_dict_type), self.image_names)

        if len(result) != len(self.images):
            raise ValueError("Thread return has not the same length as the input parameters!")

        for image_idx in self.images.keys():
            self.images[image_idx].aruco_corners = result[image_idx - 1][0]
            self.images[image_idx].aruco_id = result[image_idx - 1][1]
            self.images[image_idx].image_path = self.image_names[image_idx - 1]
            self.images[image_idx].image = cv2.resize(result[image_idx - 1][2], (0, 0), fx=0.4, fy=0.4)

    def run(self) -> np.ndarray:
        """
        Starts the aruco extraction, ray casting and intersection of lines.

        :return:
        """
        self.detect()
        self.__ray_cast()
        self.aruco_corners_3d = intersect_parallelized(P0=self.P0.reshape(len(self.P0) // 3, 3),
                                                       N=self.N.reshape(len(self.N) // 12, 4, 3))
        self.aruco_distance = self.evaluate()

        return self.aruco_distance

    def analyze(self):

        distance = np.zeros(len(self.P0) // 3)

        for i in range(2, len(self.P0) // 3 + 1):
            P0_i = self.P0.reshape(len(self.P0) // 3, 3)[:i]
            N_i = self.N.reshape(len(self.N) // 12, 4, 3)[:i]
            aruco_corners_3d = intersect_parallelized(P0=P0_i, N=N_i)

            aruco_distance = self.__evaluate(aruco_corners_3d)

            distance[i - 1] = aruco_distance

        plt.figure()
        plt.title('Aruco Distance over Number of Images')
        plt.xlabel('# Images')
        plt.ylabel('Aruco Distance')
        plt.grid()
        plt.plot(np.linspace(1, len(self.P0) // 3, len(self.P0) // 3)[1:], distance[1:])
        plt.show()

    @staticmethod
    def __evaluate(aruco_corners_3d: np.ndarray) -> np.ndarray:
        dist1 = np.linalg.norm(aruco_corners_3d[0] - aruco_corners_3d[1])
        dist2 = np.linalg.norm(aruco_corners_3d[1] - aruco_corners_3d[2])
        dist3 = np.linalg.norm(aruco_corners_3d[2] - aruco_corners_3d[3])
        dist4 = np.linalg.norm(aruco_corners_3d[3] - aruco_corners_3d[0])

        # Average
        return np.mean([dist1, dist2, dist3, dist4])

    def evaluate(self) -> np.ndarray:
        """
        Calculates the L2 norm between every neighbouring aruco corner. Finally the distances are averaged and returned

        :return:
        """
        # Average
        self.aruco_distance = self.__evaluate(self.aruco_corners_3d)

        return self.aruco_distance

    def apply(self, true_scale: float) -> o3d.cpu.pybind.geometry.PointCloud:
        """
        This function can be used if the scaling of the dense point cloud should be applied directly + the extrinsic
        paramters should be scaled.

        ToDo: save them to a folder!

        :param true_scale:
        :return:
        """
        self.scale_factor = (true_scale / 100) / self.aruco_distance
        self.dense_scaled = self.get_dense().__copy__().scale(scale=self.scale_factor,
                                                              center=np.asarray([0., 0., 0.]))

        # ToDo: Scale tvec and save

        return self.dense_scaled, self.scale_factor


if __name__ == '__main__':
    aruco_scale_factor = ArucoScaleFactor(project_path='data')
    aruco_distance = aruco_scale_factor.run()
    print('Mean distance between aruco markers: ', aruco_distance)

    aruco_scale_factor.analyze()

    dense, scale_factor = aruco_scale_factor.apply(true_scale=args.aruco_size)
    print('Point cloud and poses are scaled by: ', scale_factor)
