#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code for COLMAP readout borrowed from https://github.com/uzh-rpg/colmap_utils/tree/97603b0d352df4e0da87e3ce822a9704ac437933
"""

# Built-in/Generic Imports
# ...

# Libs
import pathlib as path

# Own modules
try:
    from helper.utils import *
    from helper.visualization import *
except ImportError:
    from .helper.utils import *
    from .helper.visualization import *


class COLMAP:
    def __init__(self, project_path: str, dense_pc: str = 'fused.ply'):
        '''
        This is a simple COLMAP project wrapper to simplify the readout of a COLMAP project.
        THE COLMAP project is assumed to be in the following workspace folder structure as suggested in the COLMAP
        documentation (https://colmap.github.io/format.html):

            +── images
            │   +── image1.jpg
            │   +── image2.jpg
            │   +── ...
            +── sparse
            │   +── cameras.txt
            │   +── images.txt
            │   +── points3D.txt
            +── stereo
            │   +── consistency_graphs
            │   │   +── image1.jpg.photometric.bin
            │   │   +── image2.jpg.photometric.bin
            │   │   +── ...
            │   +── depth_maps
            │   │   +── image1.jpg.photometric.bin
            │   │   +── image2.jpg.photometric.bin
            │   │   +── ...
            │   +── normal_maps
            │   │   +── image1.jpg.photometric.bin
            │   │   +── image2.jpg.photometric.bin
            │   │   +── ...
            │   +── patch-match.cfg
            │   +── fusion.cfg
            +── fused.ply
            +── meshed-poisson.ply
            +── meshed-delaunay.ply
            +── run-colmap-geometric.sh
            +── run-colmap-photometric.sh

        :param project_path:
        :param image_path:
        '''
        self.__project_path = path.Path(project_path)
        self.__src_image_path = self.__project_path.joinpath('images')
        self.__sparse_base_path = self.__project_path.joinpath('sparse')
        self.__camera_path = self.__sparse_base_path.joinpath('cameras.bin')
        self.__image_path = self.__sparse_base_path.joinpath('images.bin')
        self.__points3D_path = self.__sparse_base_path.joinpath('points3D.bin')
        self.__fused_path = self.__project_path.joinpath(dense_pc)

        self.__read_cameras()
        self.__read_images()
        self.__read_sparse_model()
        self.__read_dense_model()

    def __read_cameras(self):
        self.cameras = read_cameras_binary(self.__camera_path)

    def __read_images(self):
        self.images = read_images_binary(self.__image_path)

    def __read_sparse_model(self):
        self.sparse = read_points3d_binary(self.__points3D_path)

    def __read_dense_model(self):
        self.dense = o3d.io.read_point_cloud(self.__fused_path.__str__())

    def get_sparse(self):
        return generate_colmap_sparse_pc(self.sparse)

    def show_sparse(self):
        sparse = self.get_sparse()
        o3d.visualization.draw_geometries([sparse])

    def get_dense(self):
        return self.dense

    def show_dense(self):
        dense = self.get_dense()
        o3d.visualization.draw_geometries([dense])

    def visualization(self, frustum_scale=1):
        """

        :param frustum_scale:
        :return:
        """

        geometries = []
        geometries.append(self.get_dense())

        for image_idx in self.images.keys():
            camera_intrinsics = Intrinsics(camera=self.cameras[self.images[image_idx].camera_id])

            Rwc, twc, M = convert_colmap_extrinsics(frame=self.images[image_idx])

            line_set, sphere, mesh = draw_camera_viewport(extrinsics=M,
                                                          intrinsics=camera_intrinsics.K,
                                                          image=None,
                                                          scale=frustum_scale)

            geometries.append(mesh)
            geometries.append(line_set)
            geometries.append(sphere)

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


if __name__ == '__main__':
    project = COLMAP(project_path='data')

    camera = project.cameras
    images = project.images
    sparse = project.get_sparse()
    dense = project.get_dense()

    project.visualization()
