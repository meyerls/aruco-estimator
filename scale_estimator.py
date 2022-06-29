#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
import argparse
import os

# Libs
import open3d as o3d

# Own modules
try:
    from src.aruco_estimator import ArucoScaleFactor, download, DEBUG
except ModuleNotFoundError:
    from aruco_estimator import ArucoScaleFactor, download, DEBUG

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate scale factor for COLMAP projects with aruco markers.')
    parser.add_argument('--colmap_project', type=str, help='Path to COLMAP project')
    parser.add_argument('--dense_model', type=str, help='name to the dense model', default='fused.ply')
    parser.add_argument('--aruco_size', type=float, help='Size of the aruco marker in cm.', default=15)
    parser.add_argument('--visualize', action='store_true', help='Flag to enable visualization')
    parser.add_argument('--point_size', type=float, help='Point size of the visualized dense point cloud. '
                                                         'Depending on the number of points in the model. '
                                                         'Between 0.001 and 2', default=2)
    parser.add_argument('--frustum_size', type=float, help='Size of the visualized camera frustums. '
                                                           'Between 0 (small) and 1 (large)', default=0.7)
    parser.add_argument('--test_data', action='store_true', help='Download and try out test data')
    args = parser.parse_args()

    if args.test_data:
        # Download example dataset. Door dataset is roughly 200 MB
        dataset = download.Dataset()
        dataset.download_door_dataset()

        args.colmap_project = dataset.dataset_path
        args.aruco_size = dataset.scale

    if isinstance(args.colmap_project, type(None)):
        raise ValueError('--colmap_project is empty! Please select a path to our colmap project or test it with our '
                         'dataset by setting the flag --test_data')

    # Init & run pose estimation of corners in 3D & estimate mean L2 distance between the four aruco corners
    aruco_scale_factor = ArucoScaleFactor(project_path=args.colmap_project, dense_path=args.dense_model)
    aruco_distance = aruco_scale_factor.run()
    print('Mean distance between aruco markers: ', aruco_distance)

    # Calculate scaling factor and apply to scene
    dense, scale_factor = aruco_scale_factor.apply(true_scale=args.aruco_size)
    print('Point cloud and poses are scaled by: ', scale_factor)

    if DEBUG:
        aruco_scale_factor.visualize_scaled_scene()

    # Visualization of the scene and rays BEFORE scaling. This might be necessary for debugging
    if args.visualize:
        aruco_scale_factor.visualize_estimation(frustum_scale=args.frustum_size, point_size=args.point_size)

    # aruco_scale_factor._ArucoScaleFactor__visualization_scaled_scene(frustum_scale=0.2)
    # Todo: Save output, PCD and poses. Visualize!
    o3d.io.write_point_cloud(os.path.join(args.colmap_project, 'scaled.ply'), dense)
    aruco_scale_factor.write_camera_data()
