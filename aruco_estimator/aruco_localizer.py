#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

import logging
import os
from functools import partial
from multiprocessing import Pool
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .opt import (
    intersect_parallelized,
)
from .sfm.colmap import COLMAPProject
from .utils import ray_cast_aruco_corners


def detect_aruco_marker(
    image: np.ndarray,
    dict_type: int = cv2.aruco.DICT_4X4_100,
    aruco_parameters: cv2.aruco.DetectorParameters = cv2.aruco.DetectorParameters(),
) -> Tuple[tuple, np.ndarray]:
    """
    Info: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
    More information on aruco parameters: https://docs.opencv.org/4.x/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html

    @param dict_type:
    @param image:
    @param aruco_parameters:

    Aruco Corners

        p1------------p2
        |             |
        |             |
        |             |
        |             |
        p4------------p3

    :param image:
    :return:
    """

    image = cv2.imread(image)
    if image is None:
        logging.warning(f"Failed to load image: {image}")
        return None, None
    image_size = image.shape
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    a = dict({aruco_dict: 23})
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_parameters)
    corners, aruco_id, _ = detector.detectMarkers(image)
    if aruco_id is None:
        return None, None, image_size
    return corners, aruco_id, image_size
class ArucoLocalizer():
    def __init__(
        self,
        project: COLMAPProject,
        dict_type: int = cv2.aruco.DICT_4X4_50,
        target_id: int = 0,
    ):
        """
        This class is used to determine 3D points of the aruco marker, which are used to compute a scaling factor.
        In the following the workflow is shortly described.

                  ---------------------             Load data from SfM project. They include extrinsic, intrinsic
                  |  Load SfM Data     |             parameters, images, sparse and dense point cloud
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
                | LS for Intersection of Lines |
                -------------------------------

        :param project: COLMAPProject instance (or any SfmProjectBase implementation)
        :param dict_type: ArUco dictionary type
        :param target_id: Target ArUco ID to use as origin (default: 0)
        """
        self.project = project
        self.aruco_marker_detected = None

        # Values to calculate 3D point of intersection
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

        # Multi Processing
        self.progress_bar = True
        self.num_processes = 12 if os.cpu_count() > 12 else os.cpu_count()
        logging.debug("Num process: %s", self.num_processes)
        
        # Prepare image paths for processing
        self.image_names = []
        for image_id, image in self.project.images.items():
            full_path = os.path.join(self.project.images_path, image.name)
            self.image_names.append(full_path)

        
        # Add attributes to store ArUco detection results
        self._init_image_attributes()
        
    def _init_image_attributes(self):
        """Initialize additional attributes for images to store ArUco data."""
        for image_id, image in self.project.images.items():
            # We'll store these as separate dictionaries since namedtuples are immutable
            if not hasattr(self, 'image_aruco_data'):
                self.image_aruco_data = {}
            
            self.image_aruco_data[image_id] = {
                'aruco_corners': None,
                'aruco_id': None,
                'p0': None,
                'n': None,
                'image_path': None
            }
        
    @staticmethod
    def __evaluate(aruco_corners_3d: np.ndarray) -> np.ndarray:
        """
        Calculates the L2 norm between every neighbouring aruco corner. 
        Finally the distances are averaged and returned
        
        @param aruco_corners_3d: 4x3 array of corner positions
        @return: Average distance between adjacent corners
        """
        dist1 = np.linalg.norm(aruco_corners_3d[0] - aruco_corners_3d[1])
        dist2 = np.linalg.norm(aruco_corners_3d[1] - aruco_corners_3d[2])
        dist3 = np.linalg.norm(aruco_corners_3d[2] - aruco_corners_3d[3])
        dist4 = np.linalg.norm(aruco_corners_3d[3] - aruco_corners_3d[0])

        # Average
        return np.mean([dist1, dist2, dist3, dist4])

    def run(self) -> Tuple[float, np.ndarray]:
        """
        Starts the aruco extraction, ray casting and intersection of lines.

        :return: Tuple of (aruco_distance, aruco_corners_3d) for the target ArUco marker
        """
        self.__detect()
        self.__ray_cast()

        # Check if target ID was found
        if self.target_id not in self.all_aruco_ids:
            raise ValueError(
                f"Target ArUco ID {self.target_id} not found in images. Available IDs: {self.all_aruco_ids}"
            )

        # Calculate 3D corners for target ID
        self.aruco_corners_3d = intersect_parallelized(
            P0=self.P0.reshape(len(self.P0) // 3, 3),
            N=self.N.reshape(len(self.N) // 12, 4, 3),
        )
        self.aruco_distance = self.__evaluate(self.aruco_corners_3d)

        # Store the target corners in the dictionary
        self.all_aruco_corners_3d[self.target_id] = self.aruco_corners_3d

        # Calculate positions for all detected ArUco markers
        self.calculate_all_aruco_positions()

        return self.aruco_distance, self.aruco_corners_3d

    def __detect(self):
        """
        Detects the aruco corners in the image and extracts the aruco id.
        """
        with Pool(self.num_processes) as p:
            result = list(
                tqdm(
                    p.imap(
                        partial(detect_aruco_marker, dict_type=self.aruco_dict_type),
                        self.image_names,
                    ),
                    total=len(self.image_names),
                    disable=not self.progress_bar,
                )
            )

        if len(result) != len(self.project.images):
            raise ValueError(
                "Thread return has not the same length as the input parameters!"
            )

        aruco_ids = []
        self.all_aruco_ids = []  # Reset the list of all ArUco IDs

        for image_idx, image_id in enumerate(self.project.images.keys()):
            image = self.project.images[image_id]
            
            # Get camera parameters
            camera_id = image.camera_id
            camera = self.project.cameras[camera_id]
            
            # Calculate scaling ratios
            ratio_x = camera.width / result[image_idx][2][1]
            ratio_y = camera.height / result[image_idx][2][0]
            
            if result[image_idx][0] is not None:
                corners = (
                    np.expand_dims(
                        np.vstack(
                            [
                                result[image_idx][0][0][0, :, 0] * ratio_y,
                                result[image_idx][0][0][0, :, 1] * ratio_x,
                            ]
                        ).T,
                        axis=0,
                    ),
                )
                self.image_aruco_data[image_id]['aruco_corners'] = corners

                # Store the ArUco ID
                current_id = result[image_idx][1][0][0]
                aruco_ids.append(current_id)

                # Add to the list of all ArUco IDs if not already present
                if current_id not in self.all_aruco_ids:
                    self.all_aruco_ids.append(current_id)
            else:
                self.image_aruco_data[image_id]['aruco_corners'] = result[image_idx][0]

            self.image_aruco_data[image_id]['aruco_id'] = result[image_idx][1]
            self.image_aruco_data[image_id]['image_path'] = self.image_names[image_idx]

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
        This function casts a ray from the camera center through the detected aruco corners.
        """
        # Reset P0 and N for the target ID
        self.P0 = np.array([])
        self.N = np.array([])

        # Dictionary to store P0 and N for each ArUco ID
        self.all_P0 = {id: np.array([]) for id in self.all_aruco_ids}
        self.all_N = {id: np.array([]) for id in self.all_aruco_ids}

        for image_id in self.project.images.keys():
            if self.image_aruco_data[image_id]['aruco_corners'] is not None:
                current_id = self.image_aruco_data[image_id]['aruco_id'][0, 0]
                
                # Get image and camera data
                image = self.project.images[image_id]
                camera = self.project.cameras[image.camera_id]

                # Process all detected ArUco markers using the standardized interface
                p0, n = ray_cast_aruco_corners(
                    extrinsics=image.world_extrinsics,
                    intrinsics=camera.intrinsics.K,
                    corners=self.image_aruco_data[image_id]['aruco_corners'],
                )

                self.image_aruco_data[image_id]['p0'] = p0
                self.image_aruco_data[image_id]['n'] = n

                # Store data for all ArUco IDs
                self.all_P0[current_id] = np.append(self.all_P0[current_id], p0)
                self.all_N[current_id] = np.append(self.all_N[current_id], n)

                # If this is the target ID, also store in the main P0 and N arrays
                if current_id == self.target_id:
                    self.P0 = np.append(self.P0, p0)
                    self.N = np.append(self.N, n)

    def calculate_all_aruco_positions(self):
        """
        Calculate 3D positions for all detected ArUco markers.

        :return: Dictionary mapping ArUco IDs to their 3D corner positions
        """
        # Calculate 3D corners for all ArUco IDs
        for aruco_id in self.all_aruco_ids:
            if len(self.all_P0[aruco_id]) > 0:  # Only process if we have data for this ID
                P0_reshaped = self.all_P0[aruco_id].reshape(
                    len(self.all_P0[aruco_id]) // 3, 3
                )
                N_reshaped = self.all_N[aruco_id].reshape(
                    len(self.all_N[aruco_id]) // 12, 4, 3
                )

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

