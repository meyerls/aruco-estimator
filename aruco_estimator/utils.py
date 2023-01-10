#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

import numpy as np
import open3d as o3d
from copy import copy


def kabsch_umeyama(A, B):
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t


def align_point_set(point_set_A, point_set_B):
    R, c, t = kabsch_umeyama(np.asarray(point_set_A), np.asarray(point_set_B))

    point_set_B = np.array([t + c * R @ b for b in point_set_B])

    return point_set_A, point_set_B, [R, c, t]


def plot_aligned_pointset(A, B):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(A)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(B)

    o3d.visualization.draw_geometries([pcd1, pcd2])


def get_icp_transformation(source, target, trafo, max_iteration=2000):
    threshold = 0.02
    trans_init = np.eye(4)
    trans_init[:3, :4] = np.hstack([trafo[1] * trafo[0], np.expand_dims(trafo[2], axis=0).T])

    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

    return reg_p2p


def manual_registration(pcd_1, pcd_2):
    def pick_points(pcd):
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name='Picker')
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        return vis.get_picked_points()

    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(pcd_1)
    picked_id_target = pick_points(pcd_2)

    picked_points_1 = pcd_1.select_by_index(picked_id_source)
    picked_points_2 = pcd_1.select_by_index(picked_id_target)

    return np.asarray(picked_points_1.points), np.asarray(picked_points_2.points)
