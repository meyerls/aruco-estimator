#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
from typing import Tuple

# Libs
import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion

# Own modules
# ...


def text_3d(text, pos, direction=None, density=10, degree=0.0, font="arial.ttf", font_size=16):
    """
    Source: https://github.com/isl-org/Open3D/issues/2

    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: radius of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    font_obj = ImageFont.truetype(font, font_size * density)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 1000.0 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


def create_sphere_mesh(t: np.ndarray, color: list, radius: float) -> o3d.cpu.pybind.geometry.TriangleMesh:
    '''
    Creates a sphere mesh, is translated to a parsed 3D coordinate and has uniform color

    :param t: 3D Coordinate
    :param color: rgb color ranging between 0 and 1.
    :param radius: radius of the sphere
    :return:
    '''
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(t)
    sphere.paint_uniform_color(np.asarray(color))

    return sphere


def generate_line_set(points: list, lines: list, color: list) -> o3d.cpu.pybind.geometry.LineSet:
    '''
    Generates a line set of parsed points, with uniform color.

    :param points: points of lines
    :param lines: list of connections
    :param color: rgb color ranging between 0 and 1.
    :return:
    '''
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def draw_camera_plane(extrinsics, intrinsics, scale):
    '''
    Draw camera/image plane inside the camera frustum.

    :param extrinsics:
    :param intrinsics:
    :param scale:
    :return:
    '''

    # Extrinsic parameters
    R, t = extrinsics[:3, :3], extrinsics[:3, 3]

    # intrinsic points
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

    # Normalize to 1
    max_norm = max(fx, fy, cx, cy)

    # Define plane corner points
    points_camera_plane = [
        t + (np.asarray([cx, cy, fx]) * scale) / max_norm @ R.T,
        t + (np.asarray([cx, -cy, fx]) * scale) / max_norm @ R.T,
        t + (np.asarray([-cx, -cy, fx]) * scale) / max_norm @ R.T,
        t + (np.asarray([-cx, cy, fx]) * scale) / max_norm @ R.T,
    ]

    # Define line connections
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], ]
    line_set = generate_line_set(points=points_camera_plane,
                                 lines=lines,
                                 color=[1, 0, 0])

    return line_set, points_camera_plane


def draw_camera_viewport(extrinsics: np.ndarray, intrinsics: np.ndarray, image=None, scale=1) \
        -> Tuple[
            o3d.cpu.pybind.geometry.LineSet,
            o3d.cpu.pybind.geometry.TriangleMesh,
            o3d.cpu.pybind.geometry.TriangleMesh]:
    '''

    :param extrinsics:
    :param intrinsics:
    :param image_path:
    :param scale:
    :return:
    '''

    # Extrinsic parameters
    R, t = extrinsics[:3, :3], extrinsics[:3, 3]

    # intrinsic points
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

    # Normalize to 1
    max_norm = max(fx, fy, cx, cy)

    points = [
        t,
        t + (np.asarray([cx, cy, fx]) * scale) / max_norm @ R.T,
        t + (np.asarray([cx, -cy, fx]) * scale) / max_norm @ R.T,
        t + (np.asarray([-cx, -cy, fx]) * scale) / max_norm @ R.T,
        t + (np.asarray([-cx, cy, fx]) * scale) / max_norm @ R.T,
    ]

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    line_set = generate_line_set(points=points, lines=lines, color=[1, 0, 0])

    sphere = create_sphere_mesh(t=t, color=[1, 0, 0], radius=0.01)

    # Fill image plane/mesh with image as texture
    if isinstance(image, np.ndarray):

        # Create Point Cloud and assign a normal vector pointing in the opposite direction of the viewing normal
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points[1:]))
        normal_vec = - (np.asarray([0, 0, 1]) @ R.T)
        pcd.normals = o3d.utility.Vector3dVector(np.tile(normal_vec, (pcd.points.__len__(), 1)))

        # Create image plane with image as texture
        plane = o3d.geometry.TriangleMesh()
        plane.vertices = pcd.points
        plane.triangles = o3d.utility.Vector3iVector(np.asarray([[0, 1, 3], [1, 2, 3]]))
        plane.compute_vertex_normals()
        v_uv = np.asarray([[1, 1], [1, 0], [0, 1], [1, 0], [0, 0], [0, 1]])
        plane.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
        plane.triangle_material_ids = o3d.utility.IntVector([0] * 2)
        plane.textures = [o3d.geometry.Image(cv2.resize(image, (0, 0), fx=0.4, fy=0.4))]

        # mesh = draw_image2camera_mesh(extrinsics=extrinsics, intrinsics=intrinsics, image=image, scale=scale)
    else:
        plane = o3d.geometry.TriangleMesh()

    return line_set, sphere, plane
