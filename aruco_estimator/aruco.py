#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ArUco marker detection and 3D localization functions.
"""

import logging
import os
from functools import partial
from multiprocessing import Pool
from typing import Dict, Tuple, List

import cv2
import numpy as np
from tqdm import tqdm
from .opt import intersect_parallelized


def ray_cast_aruco_corners(
    extrinsics: np.ndarray, intrinsics: np.ndarray, corners: tuple
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cast rays from camera center through ArUco corners.
    
    :param extrinsics: Camera extrinsics matrix
    :param intrinsics: Camera intrinsics matrix 
    :param corners: ArUco corner coordinates
    :return: (camera_origin, ray_directions)
    """
    R, camera_origin = extrinsics[:3, :3], extrinsics[:3, 3]
    aruco_corners = np.concatenate((corners[0][0], np.ones((4, 1))), axis=1)
    rays = aruco_corners @ np.linalg.inv(intrinsics).T @ R.T
    rays_norm = rays / np.linalg.norm(rays, ord=2, axis=1, keepdims=True)
    return camera_origin, rays_norm


def detect_aruco_markers_in_image(
    image: np.ndarray, dict_type: int, detector_params: cv2.aruco.DetectorParameters
) -> Tuple[tuple, np.ndarray, tuple]:
    """
    Detect ArUco markers in a single image.
    
    :param image: Input image
    :param dict_type: ArUco dictionary type
    :param detector_params: Detector parameters
    :return: (corners, aruco_ids, image_size)
    """
    # Create detector inside worker process (for multiprocessing compatibility)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    
    image_size = image.shape
    corners, aruco_ids, _ = detector.detectMarkers(image)
    if aruco_ids is None:
        return None, None, image_size
    return corners, aruco_ids, image_size


def localize_aruco_markers(
    project, 
    dict_type: int,
    detector: cv2.aruco.ArucoDetector,
    progress_bar: bool = True,
    num_processes: int = None
) -> Dict[int, Dict]:
    """
    Detect and localize ArUco markers in 3D space for a single dictionary type.
    
    :param project: SfmProjectBase instance
    :param dict_type: ArUco dictionary type identifier
    :param detector: Configured ArucoDetector
    :param progress_bar: Show progress bars
    :param num_processes: Number of processes for multiprocessing
    :return: Dictionary mapping aruco_id -> marker_data
    """
    if num_processes is None:
        num_processes = min(12, os.cpu_count())
    
    # Disable multiprocessing if only 1 process or if it causes issues
    if num_processes <= 1:
        num_processes = None
    
    logging.info(f"Processing ArUco dictionary type: {dict_type}")
    
    # Detect markers in all images
    detection_data = _detect_markers_in_project(
        project, dict_type, detector, progress_bar, num_processes
    )
    
    # Calculate 3D positions
    marker_results = _calculate_3d_positions(
        project, dict_type, detection_data
    )
    
    if marker_results:
        logging.info(f"Dict {dict_type}: Found {len(marker_results)} markers")
    else:
        logging.warning(f"Dict {dict_type}: No markers found")
    
    return marker_results


def _detect_markers_in_project(
    project, dict_type: int, detector: cv2.aruco.ArucoDetector,
    progress_bar: bool, num_processes: int
) -> Dict:
    """
    Detect ArUco markers in all project images.
    
    :return: Detection data organized by image_id
    """
    # Load images
    image_ids = list(project.images.keys())
    images = []
    
    for image_id in tqdm(image_ids, desc=f"Loading images for dict {dict_type}", 
                        disable=not progress_bar):
        image = project.load_image_by_id(image_id)
        images.append(image)

    # Try multiprocessing if num_processes > 1, fall back to sequential if it fails
    if num_processes and num_processes > 1:
        try:
            # Extract detector parameters for multiprocessing
            detector_params = detector.getDetectorParameters()
            
            # Detect markers using multiprocessing
            with Pool(num_processes) as p:
                results = list(
                    tqdm(
                        p.imap(
                            partial(detect_aruco_markers_in_image, 
                                   dict_type=dict_type, 
                                   detector_params=detector_params),
                            images,
                        ),
                        total=len(images),
                        disable=not progress_bar,
                        desc=f"Detecting ArUco (dict {dict_type}) - multiprocessing"
                    )
                )
        except Exception as e:
            logging.warning(f"Multiprocessing failed ({e}), falling back to sequential processing")
            results = _detect_markers_sequential(images, detector, dict_type, progress_bar)
    else:
        # Sequential processing
        results = _detect_markers_sequential(images, detector, dict_type, progress_bar)

    # Process detection results
    detection_data = {}
    detected_ids = []
    
    for image_idx, image_id in enumerate(image_ids):
        image = project.images[image_id]
        camera = project.cameras[image.camera_id]
        result = results[image_idx]
        
        # Calculate scaling ratios
        ratio_x = camera.width / result[2][1]
        ratio_y = camera.height / result[2][0]
        
        detection_data[image_id] = {
            'aruco_corners': [],
            'aruco_ids': [],
            'corner_pixels': []  # Store original pixel coordinates
        }
        
        if result[0] is not None and result[1] is not None:
            for corner_set, marker_id in zip(result[0], result[1]):
                # Scale corners to camera resolution
                scaled_corners = (
                    np.expand_dims(
                        np.vstack([
                            corner_set[0, :, 0] * ratio_y,
                            corner_set[0, :, 1] * ratio_x,
                        ]).T,
                        axis=0,
                    ),
                )
                
                detection_data[image_id]['aruco_corners'].append(scaled_corners)
                detection_data[image_id]['aruco_ids'].append(marker_id[0])
                detection_data[image_id]['corner_pixels'].append(corner_set[0])
                detected_ids.append(marker_id[0])

    unique_ids = list(set(detected_ids))
    if unique_ids:
        logging.info(f"Dict {dict_type}: Detected ArUco IDs: {sorted(unique_ids)}")
    else:
        logging.warning(f"Dict {dict_type}: No ArUco markers detected")
    
    return detection_data


def _detect_markers_sequential(images, detector, dict_type, progress_bar):
    """
    Sequential marker detection (fallback when multiprocessing fails).
    """
    results = []
    for image in tqdm(images, 
                     desc=f"Detecting ArUco (dict {dict_type}) - sequential", 
                     disable=not progress_bar):
        image_size = image.shape
        corners, aruco_ids, _ = detector.detectMarkers(image)
        if aruco_ids is None:
            results.append((None, None, image_size))
        else:
            results.append((corners, aruco_ids, image_size))
    return results



def _calculate_3d_positions(project, dict_type: int, detection_data: Dict) -> Dict:
    """
    Calculate 3D positions for detected ArUco markers.
    
    :return: Dictionary mapping aruco_id to marker data
    """
    # Collect all unique ArUco IDs
    all_ids = set()
    for image_data in detection_data.values():
        all_ids.update(image_data['aruco_ids'])

    # Collect ray casting data for each marker
    ray_data = {aruco_id: {'P0': [], 'N': [], 'image_ids': [], 'corner_pixels': []} 
                for aruco_id in all_ids}

    # Process each image
    for image_id, image_data in detection_data.items():
        if not image_data['aruco_corners']:
            continue
            
        image = project.images[image_id]
        camera = project.cameras[image.camera_id]

        # Process each detected marker in this image
        for corners, aruco_id, pixels in zip(
            image_data['aruco_corners'], 
            image_data['aruco_ids'],
            image_data['corner_pixels']
        ):
            # Cast rays for this marker
            p0, n = ray_cast_aruco_corners(
                extrinsics=image.world_extrinsics,
                intrinsics=camera.intrinsics.K,
                corners=corners,
            )

            ray_data[aruco_id]['P0'].append(p0)
            ray_data[aruco_id]['N'].append(n)
            ray_data[aruco_id]['image_ids'].append(image_id)
            ray_data[aruco_id]['corner_pixels'].append(pixels)

    # Calculate 3D positions
    marker_results = {}
    
    for aruco_id, data in ray_data.items():
        if len(data['P0']) > 0:
            # Convert to numpy arrays and reshape
            P0_array = np.array(data['P0'])
            N_array = np.array(data['N'])
            
            P0_reshaped = P0_array.reshape(-1, 3)
            N_reshaped = N_array.reshape(-1, 4, 3)

            if len(P0_reshaped) > 0 and len(N_reshaped) > 0:
                corners_3d = intersect_parallelized(P0=P0_reshaped, N=N_reshaped)
                center_xyz = np.mean(corners_3d, axis=0)
                
                marker_results[aruco_id] = {
                    'corners_3d': corners_3d,
                    'center_xyz': center_xyz,
                    'detection_count': len(P0_reshaped),
                    'image_ids': data['image_ids'],
                    'corner_pixels': data['corner_pixels'],
                    'dict_type': dict_type
                }
                
                logging.info(f"Dict {dict_type}, ID {aruco_id}: "
                            # f"error={error:.4f}, "
                           f"detections={len(P0_reshaped)}")

    return marker_results