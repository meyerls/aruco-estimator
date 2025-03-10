#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""
import logging
import os
import time
from copy import deepcopy
from functools import partial, wraps
from multiprocessing import Pool
from typing import Tuple, Union

import cv2
import numpy as np
import open3d as o3d
from colmap_wrapper.colmap import COLMAP, COLMAPProject
from colmap_wrapper.colmap.bin import (
    write_cameras_text,
    write_images_text,
    write_points3D_text,
)
from colmap_wrapper.colmap.utils import generate_colmap_sparse_pc
from tqdm import tqdm

from ..aruco import detect_aruco_marker, ray_cast_aruco_corners
from ..opt import (
    intersect,
    intersect_parallelized,
    ls_intersection_of_lines,
    ls_intersection_of_lines_parallelized,
)
from .base import LocalizerBase


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logging.debug(f'Function {func.__name__}{args} {kwargs} took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


class ArucoLocalizer(LocalizerBase):
    def __init__(self, photogrammetry_software: Union[COLMAPProject, COLMAP], aruco_size: float,
                 dense_path: str = 'fused.ply', dict_type: int = cv2.aruco.DICT_4X4_50, target_id: int = 0):
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

        :param photogrammetry_software: COLMAP project or instance
        :param aruco_size: Size of the ArUco marker in meters
        :param dense_path: Path to dense point cloud
        :param dict_type: ArUco dictionary type
        :param target_id: Target ArUco ID to use as origin (default: 0)
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
        self.aruco_dict_type = dict_type
        self.target_id = target_id
        
        # Store all detected ArUco markers
        self.all_aruco_ids = []
        self.all_aruco_corners_3d = {}  # Dictionary mapping ArUco IDs to their 3D corner positions

        # Results
        self.scale_factor = None
        self.dense_scaled = None
        self.sparse_scaled = None

        # Multi Processing
        self.progress_bar = True
        self.num_processes = 12 if os.cpu_count() > 12 else os.cpu_count()
        logging.debug('Num process: %s', self.num_processes)
        self.image_names = []
        # Prepare parsed data for multi processing
        for image_idx in self.photogrammetry_software.images.keys():
            self.image_names.append(self.photogrammetry_software._src_image_path.joinpath(
                self.photogrammetry_software.images[image_idx].name).__str__())
   
        self.aruco_size = aruco_size

    def run(self) -> [np.ndarray, None]:
        """
        Starts the aruco extraction, ray casting and intersection of lines.

        :return: Tuple of (aruco_distance, aruco_corners_3d) for the target ArUco marker
        """
        self.__detect()
        self.__ray_cast()
        
        # Check if target ID was found
        if self.target_id not in self.all_aruco_ids:
            raise ValueError(f"Target ArUco ID {self.target_id} not found in images. Available IDs: {self.all_aruco_ids}")
        
        # Calculate 3D corners for target ID
        self.aruco_corners_3d = intersect_parallelized(P0=self.P0.reshape(len(self.P0) // 3, 3),
                                                       N=self.N.reshape(len(self.N) // 12, 4, 3))
        self.aruco_distance = self.__evaluate(self.aruco_corners_3d)
        
        # Store the target corners in the dictionary
        self.all_aruco_corners_3d[self.target_id] = self.aruco_corners_3d
        
        # Calculate positions for all detected ArUco markers
        self.calculate_all_aruco_positions()

        return self.aruco_distance, self.aruco_corners_3d

    @timeit
    def __detect(self):
        """
        Detects the aruco corners in the image and extracts the aruco id. Afterwards the image, id and tuple of corners
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

        aruco_ids = []
        self.all_aruco_ids = []  # Reset the list of all ArUco IDs

        for image_idx, image_key in enumerate(self.photogrammetry_software.images.keys()):
            ratio_x = self.photogrammetry_software.cameras[1].width / result[image_idx][2][1]
            ratio_y = self.photogrammetry_software.cameras[1].height / result[image_idx][2][0]
            if result[image_idx][0] != None:
                corners = (np.expand_dims(np.vstack([result[image_idx][0][0][0, :, 0] * ratio_y,
                                                     result[image_idx][0][0][0, :, 1] * ratio_x]).T, axis=0),)
                self.photogrammetry_software.images[image_key].aruco_corners = corners
                
                # Store the ArUco ID
                current_id = result[image_idx][1][0][0]
                aruco_ids.append(current_id)
                
                # Add to the list of all ArUco IDs if not already present
                if current_id not in self.all_aruco_ids:
                    self.all_aruco_ids.append(current_id)
            else:
                self.photogrammetry_software.images[image_key].aruco_corners = result[image_idx][0]

            self.photogrammetry_software.images[image_key].aruco_id = result[image_idx][1]
            self.photogrammetry_software.images[image_key].image_path = self.image_names[image_idx]

        # Determine the dominant ArUco ID (most frequent)
        if aruco_ids:
            self.dominant_aruco_id = np.argmax(np.bincount(aruco_ids))
            
            # If target_id is not specified (or is -1), use the dominant ID
            if self.target_id == -1:
                self.target_id = self.dominant_aruco_id
                
            logging.info(f"Detected ArUco IDs: {sorted(self.all_aruco_ids)}")
            logging.info(f"Dominant ArUco ID: {self.dominant_aruco_id}")
            logging.info(f"Target ArUco ID: {self.target_id}")
        else:
            logging.warning("No ArUco markers detected in any images")

    def __ray_cast(self):
        """
        This function casts a ray from the origin of the camera center C_i (also the translational part of the extrinsic
        matrix) through the detected aruco corners in the image coordinates x = (x,y,1) which direction is defined
        trough n = x @ K^-1 @ R.T. K^-1 is the inverse of the intrinsic matrix and R is the extrinsic matrix
        (only rotation) of the current frame. Afterwards the origin C_i and the normalized direction vector for all
        aruco corners are saved.

        :return:
        """
        # Reset P0 and N for the target ID
        self.P0 = np.array([])
        self.N = np.array([])
        
        # Dictionary to store P0 and N for each ArUco ID
        self.all_P0 = {id: np.array([]) for id in self.all_aruco_ids}
        self.all_N = {id: np.array([]) for id in self.all_aruco_ids}
        
        for image_idx in self.photogrammetry_software.images.keys():
            if self.photogrammetry_software.images[image_idx].aruco_corners is not None:
                current_id = self.photogrammetry_software.images[image_idx].aruco_id[0, 0]
                
                # Process all detected ArUco markers
                p0, n = ray_cast_aruco_corners(extrinsics=self.photogrammetry_software.images[image_idx].extrinsics,
                                               intrinsics=self.photogrammetry_software.images[image_idx].intrinsics.K,
                                               corners=self.photogrammetry_software.images[image_idx].aruco_corners)
                
                self.photogrammetry_software.images[image_idx].p0 = p0
                self.photogrammetry_software.images[image_idx].n = n
                
                # Store data for all ArUco IDs
                self.all_P0[current_id] = np.append(self.all_P0[current_id], p0)
                self.all_N[current_id] = np.append(self.all_N[current_id], n)
                
                # If this is the target ID, also store in the main P0 and N arrays
                if current_id == self.target_id:
                    self.P0 = np.append(self.P0, p0)
                    self.N = np.append(self.N, n)

    @staticmethod
    def __evaluate(aruco_corners_3d: np.ndarray) -> np.ndarray:
        """
        Calculates the L2 norm between every neighbouring aruco corner. Finally the distances are averaged and returned


        @param aruco_corners_3d:
        @return:
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
        return self.photogrammetry_software.dense_scaled

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

        self.sparse_scaled = deepcopy(self.photogrammetry_software.sparse)
        for num in self.photogrammetry_software.sparse.keys():
            self.sparse_scaled[num].xyz = self.sparse_scaled[num].xyz * self.scale_factor

        # ToDo: Scale tvec and save
        self.photogrammetry_software.images_scaled = deepcopy(self.photogrammetry_software.images)
        for idx in self.photogrammetry_software.images_scaled.keys():
            self.photogrammetry_software.images_scaled[idx].tvec = self.photogrammetry_software.images[
                                                                       idx].tvec * self.scale_factor

        return self.photogrammetry_software.dense_scaled, self.scale_factor

    def calculate_all_aruco_positions(self):
        """
        Calculate 3D positions for all detected ArUco markers.
        
        :return: Dictionary mapping ArUco IDs to their 3D corner positions
        """
        # Calculate 3D corners for all ArUco IDs
        for aruco_id in self.all_aruco_ids:
            if len(self.all_P0[aruco_id]) > 0:  # Only process if we have data for this ID
                P0_reshaped = self.all_P0[aruco_id].reshape(len(self.all_P0[aruco_id]) // 3, 3)
                N_reshaped = self.all_N[aruco_id].reshape(len(self.all_N[aruco_id]) // 12, 4, 3)
                
                if len(P0_reshaped) > 0 and len(N_reshaped) > 0:
                    corners_3d = intersect_parallelized(P0=P0_reshaped, N=N_reshaped)
                    self.all_aruco_corners_3d[aruco_id] = corners_3d
                    logging.info(f"Calculated 3D corners for ArUco ID {aruco_id}")
        
        return self.all_aruco_corners_3d
    
    def get_all_aruco_positions(self):
        """
        Get the 3D positions of all detected ArUco markers.
        If positions haven't been calculated yet, calculate them.
        
        :return: Dictionary mapping ArUco IDs to their 3D corner positions
        """
        if not self.all_aruco_corners_3d:
            self.calculate_all_aruco_positions()
        return self.all_aruco_corners_3d

    def write_data(self):
        """
        Write scaled data to files.
        """
        pcd_scaled = self.photogrammetry_software._project_path
        sparse_scaled_path = self.photogrammetry_software._project_path.joinpath('sparse_scaled')
        cameras_scaled = sparse_scaled_path.joinpath('cameras.txt')
        images_scaled = sparse_scaled_path.joinpath('images.txt')
        points_scaled = sparse_scaled_path.joinpath('points3D.txt')

        sparse_scaled_path.mkdir(parents=True, exist_ok=True)

        write_cameras_text(self.photogrammetry_software.cameras, cameras_scaled)
        write_images_text(self.photogrammetry_software.images_scaled, images_scaled)
        write_points3D_text(self.sparse_scaled, points_scaled)

        o3d.io.write_point_cloud(os.path.join(pcd_scaled, 'scaled.ply'), self.photogrammetry_software.dense_scaled)

        # Save scale factor
        scale_factor_file_name = self.photogrammetry_software._project_path.joinpath('sparse_scaled/scale_factor.txt')
        np.savetxt(scale_factor_file_name, np.array([self.scale_factor]))


if __name__ == '__main__':
    from colmap_wrapper.colmap import COLMAP

    from aruco_estimator.visualization import ArucoVisualization

    project = COLMAP(project_path='../data/door', image_resize=0.4)

    aruco_localizer = ArucoLocalizer(photogrammetry_software=project, aruco_size=0.15)
    aruco_distance, aruco_points3d = aruco_localizer.run()
    logging.info('Mean distance between aruco markers: ', aruco_distance)

    aruco_localizer.analyze()

    dense, scale_factor = aruco_localizer.apply()
    logging.info('Point cloud and poses are scaled by: ', scale_factor)

    vis = ArucoVisualization(aruco_colmap=aruco_localizer)
    vis.visualization(frustum_scale=0.7, point_size=0.1)

    aruco_localizer.write_data()
