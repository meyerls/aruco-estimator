#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

import open3d as o3d
import numpy as np

from colmap_wrapper.colmap import COLMAP
from colmap_wrapper.colmap.colmap_project import PhotogrammetrySoftware
from colmap_wrapper.visualization import draw_camera_viewport
from aruco_estimator.visualization import *


class ScaleFactorExtimatorVisualization():
    def __init__(self, photogrammetry_software: PhotogrammetrySoftware):
        self.scale_factor_estimator = photogrammetry_software
        self.photogrammetry_software = photogrammetry_software.photogrammetry_software

        self.geometries = []

    def show_sparse(self):
        o3d.visualization.draw_geometries([self.photogrammetry_software.get_sparse()])

    def show_dense(self):
        o3d.visualization.draw_geometries([self.photogrammetry_software.get_dense()])


class ArucoVisualization(ScaleFactorExtimatorVisualization):
    def __init__(self, aruco_colmap: COLMAP, bg_color: np.ndarray = np.asarray([1, 1, 1])):
        super().__init__(aruco_colmap)

        self.vis_bg_color = bg_color

    def add_colmap_dense2geometrie(self):
        if np.asarray(self.photogrammetry_software.get_dense().points).shape[0] == 0:
            return False

        self.geometries.append(self.photogrammetry_software.get_dense())

        return True

    def add_colmap_sparse2geometrie(self):
        if np.asarray(self.photogrammetry_software.get_sparse().points).shape[0] == 0:
            return False

        self.geometries.append(self.photogrammetry_software.get_sparse())
        return True

    def add_colmap_frustums2geometrie(self, frustum_scale: float = 1., image_type: str = 'image'):
        """
        @param image_type:
        @type frustum_scale: object
        """
        import cv2

        geometries = []
        for image_idx in self.photogrammetry_software.images.keys():

            if image_type == 'image':
                image = self.photogrammetry_software.images[image_idx].getData(
                    self.photogrammetry_software.image_resize)
            elif image_type == 'depth_geo':
                image = self.photogrammetry_software.images[image_idx].depth_image_geometric
                min_depth, max_depth = np.percentile(image, [5, 95])
                image[image < min_depth] = min_depth
                image[image > max_depth] = max_depth
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image = (image / self.max_depth_scaler * 255).astype(np.uint8)
                image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
            elif image_type == 'depth_photo':
                image = self.photogrammetry_software.images[image_idx].depth_image_photometric
                min_depth, max_depth = np.percentile(
                    image, [5, 95])
                image[image < min_depth] = min_depth
                image[image > max_depth] = max_depth
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            line_set, sphere, mesh = draw_camera_viewport(
                extrinsics=self.photogrammetry_software.images[image_idx].extrinsics,
                intrinsics=self.photogrammetry_software.images[image_idx].intrinsics.K,
                image=image,
                scale=frustum_scale)

            geometries.append(mesh)
            geometries.append(line_set)
            geometries.extend(sphere)

        self.geometries.extend(geometries)


    def visualization(self, frustum_scale: float = 1, point_size: float = 1., sphere_size: float = 0.02):
        """

        @param frustum_scale:
        @param point_size:
        @param sphere_size:
        """

        # Add Dense & sparse Model to scene
        dense_exists = self.add_colmap_dense2geometrie()
        if not dense_exists:
            self.add_colmap_sparse2geometrie()
        # Add camera frustums to scene
        self.add_colmap_frustums2geometrie(frustum_scale=frustum_scale)

        for image_idx in self.photogrammetry_software.images.keys():

            if self.photogrammetry_software.images[image_idx].aruco_corners == None:
                aruco_line_set = generate_line_set(points=[],
                                                   lines=[],
                                                   color=[1, 0, 0])
            else:
                aruco_line_set = ray_cast_aruco_corners_visualization(
                    p_i=self.photogrammetry_software.images[image_idx].p0,
                    n_i=self.photogrammetry_software.images[image_idx].n,
                    corners3d=self.scale_factor_estimator.aruco_corners_3d)

            self.geometries.append(aruco_line_set)

        aruco_sphere = create_sphere_mesh(t=self.scale_factor_estimator.aruco_corners_3d,
                                          color=[[0, 0, 0],
                                                 [1, 0, 0],
                                                 [0, 0, 1],
                                                 [1, 1, 1]],
                                          radius=sphere_size)

        aruco_rect = generate_line_set(points=[self.scale_factor_estimator.aruco_corners_3d[0],
                                               self.scale_factor_estimator.aruco_corners_3d[1],
                                               self.scale_factor_estimator.aruco_corners_3d[2],
                                               self.scale_factor_estimator.aruco_corners_3d[3]],
                                       lines=[[0, 1], [1, 2], [2, 3], [3, 0]],
                                       color=[1, 0, 0])
        self.geometries.append(aruco_rect)
        self.geometries.extend(aruco_sphere)

        self.start_visualizer(point_size=point_size, title='Aruco Scale Factor Estimation')

    def start_visualizer(self,
                         point_size: float,
                         title: str = "Open3D Visualizer",
                         size: tuple = (1920, 1080)):
        viewer = o3d.visualization.Visualizer()
        viewer.create_window(window_name=title, width=size[0], height=size[1])

        for geometry in self.geometries:
            viewer.add_geometry(geometry)
        opt = viewer.get_render_option()
        # opt.show_coordinate_frame = True
        opt.point_size = point_size
        opt.line_width = 0.01
        opt.background_color = self.vis_bg_color
        viewer.run()
        viewer.destroy_window()


if __name__ == '__main__':
    project = COLMAP(project_path='/home/luigi/Dropbox/07_data/misc/bunny_data/reco_DocSem2',
                     dense_pc='fused.ply',
                     load_images=True,
                     image_resize=0.4)

    project_vs = ColmapVisualization(colmap=project.project_list[0])
    project_vs.visualization(frustum_scale=0.8, image_type='image')
