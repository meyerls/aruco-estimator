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
from aruco_estimator import ArucoScaleFactor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate scale factor for COLMAP projects with aruco markers.')
    parser.add_argument('--colmap_project', type=str, help='Path to COLMAP project', default= './aruco_estimator/data/tree') # ../2022_06_02/01 //  './aruco_estimator/data/'
    parser.add_argument('--aruco_size', type=float, help='Size of the aruco marker in cm.', default=15)
    parser.add_argument('--visualize', type=bool, help='Flag to enable visualization', default=True)
    args = parser.parse_args()

    # Init & run pose estimation of corners in 3D & estimate mean L2 distance between the four aruco corners
    aruco_scale_factor = ArucoScaleFactor(project_path=args.colmap_project)
    aruco_distance = aruco_scale_factor.run()
    print('Mean distance between aruco markers: ', aruco_distance)

    # Calculate scaling factor and apply to scene
    dense, scale_factor = aruco_scale_factor.apply(true_scale=args.aruco_size)
    print('Point cloud and poses are scaled by: ', scale_factor)

    # Visualization of the scene and rays BEFORE scaling. This might be necessary for debugging
    if args.visualize:
        aruco_scale_factor.visualization(frustum_scale=0.3, point_size=2, sphere_size=0.008)

    # aruco_scale_factor._ArucoScaleFactor__visualization_scaled_scene(frustum_scale=0.2)
    # Todo: Save output, PCD and poses. Visualize!
    o3d.io.write_point_cloud(os.path.join(args.colmap_project, 'scaled.ply'), dense)
