#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code for COLMAP readout borrowed from https://github.com/uzh-rpg/colmap_utils/tree/97603b0d352df4e0da87e3ce822a9704ac437933
"""

# Built-in/Generic Imports
import os
import platform
import sys

# Libs
import cv2
import numpy as np
from pathlib import Path
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# Own modules
try:
    from colmap.src.colmap import colmap
    from colmap.src.colmap import utils
    from colmap.src.colmap import camera
    from colmap.src.colmap import visualization
except ModuleNotFoundError:
    import colmap
    import utils
    import camera

isMacOS = (platform.system() == "Darwin")


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=np.asarray([0, 0, 0]))
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
    POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
    CUSTOM_PROFILE_NAME = "Custom"
    LIGHTING_PROFILES = {
        DEFAULT_PROFILE_NAME: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        POINT_CLOUD_PROFILE_NAME: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0
        },
    }

    def __init__(self):
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = gui.Color(1, 1, 1)
        self.show_skybox = False
        self.show_axes = False
        self.show_cameras = False
        self.show_images = False
        self.display_images = False

        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        self.material = self._materials[Settings.LIT]

    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        assert (self.material.shader == Settings.LIT)
        prefab = Settings.PREFAB[name]
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_COLMAP = 4
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    DEFAULT_IBL = "default"

    MATERIAL_NAMES = ["Lit", "Unlit", "Normals", "Depth"]
    MATERIAL_SHADERS = [
        Settings.LIT, Settings.UNLIT, Settings.NORMALS, Settings.DEPTH
    ]

    def __init__(self, width, height):
        self.settings = Settings()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

        self.window = gui.Application.instance.create_window(
            "Open3D", width, height)
        w = self.window  # to make the code more concise

        self.colmap_project = None

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_child(self._show_axes)

        self._show_skybox = gui.Checkbox("Show skymap")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        view_ctrls.add_child(self._show_skybox)

        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        view_ctrls.add_child(grid)

        self._settings_panel.add_child(view_ctrls)

        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        material_settings = gui.CollapsableVert("Material settings", 0,
                                                gui.Margins(em, 0, 0, 0))

        self._shader = gui.Combobox()
        self._shader.add_item(AppWindow.MATERIAL_NAMES[0])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[1])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[2])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[3])
        self._shader.set_on_selection_changed(self._on_shader)
        self._material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(self._on_material_prefab)
        self._material_color = gui.ColorEdit()
        self._material_color.set_on_value_changed(self._on_material_color)
        self._point_size = gui.Slider(gui.Slider.DOUBLE)
        self._point_size.set_limits(0.001, 5)
        self._point_size.double_value = 1.5
        self._point_size.set_on_value_changed(self._on_point_size)

        self._line_width_size = gui.Slider(gui.Slider.DOUBLE)
        self._line_width_size.set_limits(1, 20)
        self._line_width_size.double_value = 5
        self._line_width_size.set_on_value_changed(self._on_line_width_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Type"))
        grid.add_child(self._shader)
        # grid.add_child(gui.Label("Material"))
        # grid.add_child(self._material_prefab)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._material_color)
        grid.add_child(gui.Label("Point radius"))
        grid.add_child(self._point_size)
        grid.add_child(gui.Label("Line width"))
        grid.add_child(self._line_width_size)
        material_settings.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(material_settings)

        colmap_settings = gui.CollapsableVert("COLMAP settings", 0,
                                              gui.Margins(em, 0, 0, 0))

        self._show_cameras = gui.Checkbox("Show Cameras")
        self._show_cameras.set_on_checked(self._on_show_cameras)
        self._show_cameras.visible = False
        self._resize_cameras = gui.Slider(gui.Slider.DOUBLE)
        self._resize_cameras.set_limits(0., 1.)
        self._resize_cameras.double_value = .2
        self._resize_cameras.visible = False
        self._resize_cameras.set_on_value_changed(self._on_resize_camera)
        self._show_images = gui.Checkbox("Show Images")
        self._show_images.set_on_checked(self._on_show_images)
        self._show_images.visible = False

        self._pcd = gui.Combobox()
        self._pcd.set_on_selection_changed(self._on_pcd)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Cameras"))
        grid.add_child(self._show_cameras)
        grid.add_child(gui.Label("Camera Size"))
        grid.add_child(self._resize_cameras)
        grid.add_child(gui.Label("Images"))
        grid.add_child(self._show_images)
        grid.add_child(gui.Label("PCD"))
        grid.add_child(self._pcd)
        colmap_settings.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(colmap_settings)
        # ----

        self.image_panel = gui.Vert(100, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self._display_images = gui.ImageWidget()
        # self._display_images.visible = True
        r = self.window.content_rect
        width = 17 * self.window.theme.font_size
        self._display_images.frame = gui.Rect(r.get_right() - width * 2, r.get_bottom() - width * 2 / 3 * 2 + 105,
                                              width * 2,
                                              width * 2 / 3 * 2)
        self.image_panel.add_child(self._display_images)

        w.add_child(self.image_panel)

        self._scene.set_on_mouse(self.on_mouse)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)
            file_menu.add_item("Load COLMAP...", AppWindow.MENU_COLMAP)
            file_menu.add_item("Export Current Image...", AppWindow.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_COLMAP,
                                     self._on_menu_colmap)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT,
                                     self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)

        self._bg_color.color_value = self.settings.bg_color
        self._show_skybox.checked = self.settings.show_skybox
        self._show_axes.checked = self.settings.show_axes
        self._show_cameras.checked = self.settings.show_cameras
        self._show_images.checked = self.settings.show_images

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        self._material_prefab.enabled = (
                self.settings.material.shader == Settings.LIT)
        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])
        self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size
        self._line_width_size.double_value = self.settings.material.line_width

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + radius) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(r.height,
                     self._settings_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)
        self.image_panel.frame = gui.Rect(r.get_right() - width * 2, r.get_bottom() - width * 2 / 3 * 2 + 105,
                                          width * 2,
                                          width * 2 / 3 * 2)
        self._display_images.frame = gui.Rect(r.get_right() - width * 2, r.get_bottom() - width * 2 / 3 * 2 + 105,
                                              width * 2,
                                              width * 2 / 3 * 2)

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_menu_colmap(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Choose COLMAp folder to load",
                             self.window.theme)
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_colmap_dialog_done)
        self.window.show_dialog(dlg)

        self._show_cameras.visible = True

    def _on_pcd(self, name, index):
        self._scene.scene.remove_geometry("__model__")
        geometry = o3d.io.read_point_cloud(name)
        self._scene.scene.add_geometry("__model__", geometry,
                                       rendering.MaterialRecord())

        self._apply_settings()

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_shader(self, name, index):
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_material_prefab(self, name, index):
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_color(self, color):
        self.settings.material.base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_point_size(self, size):
        self.settings.material.point_size = size
        self.settings.apply_material = True
        self._apply_settings()

    def _on_line_width_size(self, size):
        self.settings.material.line_width = size
        self.settings.apply_material = True
        self._apply_settings()

    def _on_show_cameras(self, show):
        self.settings.show_cameras = show

        self._show_images.visible = show
        self._resize_cameras.visible = show

        if self.settings.show_cameras:
            self.camera_line_set = []
            self.camera_origin_mesh = []

            for idx in range(1, self.colmap_project.images.__len__() + 1):
                camera_intrinsics = camera.Intrinsics(self.colmap_project.cameras[1])
                camera_intrinsics.load_from_colmap(
                    camera=self.colmap_project.cameras[self.colmap_project.images[idx].camera_id])

                Rwc, twc, M = utils.convert_colmap_extrinsics(frame=self.colmap_project.images[idx])

                line_set, sphere, mesh = visualization.draw_camera_viewport(extrinsics=M,
                                                                            intrinsics=camera_intrinsics.K)

                self._scene.scene.add_geometry("Line_set_{}".format(idx), line_set,
                                               self.settings.material)
                self._scene.scene.add_geometry("sphere_{}".format(idx), sphere[0],
                                               self.settings.material)
                self.camera_line_set.append("Line_set_{}".format(idx))
                self.camera_origin_mesh.append("sphere_{}".format(idx))
                # self._scene.scene.add_geometry("Sphere_set_{}".format(idx), sphere,
                #                               self.settings.material)
                # self.camera_geometries.append("Sphere_set_{}".format(idx))
                # self._scene.scene.add_geometry("Mesh_set_{}".format(idx), mesh,
                #                               self.settings.material)
                # self.camera_geometries.append("Mesh_set_{}".format(idx))

        else:
            for geometry_name in self.camera_line_set:
                self._scene.scene.remove_geometry(geometry_name)

        self._apply_settings()

    def _on_resize_camera(self, size):

        for geometry_name1, geometry_name2 in zip(self.camera_line_set, self.camera_origin_mesh):
            self._scene.scene.remove_geometry(geometry_name1)
            self._scene.scene.remove_geometry(geometry_name2)

        self.camera_line_set = []
        self.camera_origin_mesh = []

        for idx in range(1, self.colmap_project.images.__len__() + 1):
            camera_intrinsics = camera.Intrinsics(self.colmap_project.cameras[1])
            camera_intrinsics.load_from_colmap(
                camera=self.colmap_project.cameras[self.colmap_project.images[idx].camera_id])

            Rwc, twc, M = utils.convert_colmap_extrinsics(frame=self.colmap_project.images[idx])

            line_set, sphere, mesh = visualization.draw_camera_viewport(extrinsics=M,
                                                                        intrinsics=camera_intrinsics.K,
                                                                        scale=size)

            self._scene.scene.add_geometry("Line_set_{}".format(idx), line_set,
                                           self.settings.material)
            self._scene.scene.add_geometry("sphere_{}".format(idx), sphere[0],
                                           self.settings.material)
            self.camera_line_set.append("Line_set_{}".format(idx))
            self.camera_origin_mesh.append("sphere_{}".format(idx))

            # self._scene.scene.add_geometry("Sphere_set_{}".format(idx), sphere,
            #                               self.settings.material)
            # self.camera_geometries.append("Sphere_set_{}".format(idx))
            # self._scene.scene.add_geometry("Mesh_set_{}".format(idx), mesh,
            #                               self.settings.material)
            # self.camera_geometries.append("Mesh_set_{}".format(idx))

    def _on_show_images(self, show):
        self.settings.show_images = show

        if self.settings.show_images:
            self.camera_mesh = []

            for idx in range(1, self.colmap_project.images.__len__() + 1):
                camera_intrinsics = camera.Intrinsics(self.colmap_project.cameras[1])
                camera_intrinsics.load_from_colmap(
                    camera=self.colmap_project.cameras[self.colmap_project.images[idx].camera_id])

                Rwc, twc, M = utils.convert_colmap_extrinsics(frame=self.colmap_project.images[idx])

                _, _, mesh = visualization.draw_camera_viewport(extrinsics=M, intrinsics=camera_intrinsics.K,
                                                                image=self.colmap_project.images[idx].image,
                                                                scale=self._resize_cameras.double_value)

                # self._scene.scene.add_geometry("Line_set_{}".format(idx), line_set,
                #                               self.settings.material)
                # self.camera_line_set.append("Line_set_{}".format(idx))
                # self._scene.scene.add_geometry("Sphere_set_{}".format(idx), sphere,
                #                               self.settings.material)
                # self.camera_geometries.append("Sphere_set_{}".format(idx))
                self._scene.scene.add_geometry("Mesh_set_{}".format(idx), mesh,
                                               self.settings.material)
                self.camera_mesh.append("Mesh_set_{}".format(idx))

        else:
            for geometry_name in self.camera_mesh:
                self._scene.scene.remove_geometry(geometry_name)

        self._apply_settings()

    def _on_load_colmap_dialog_done(self, foldername):
        self.window.close_dialog()

        for file in Path(foldername).glob('*.ply'):
            self._pcd.add_item(file.name)

        # self.colmap_project = colmap.COLMAP(project_path=foldername, dense_pc='cropped_02.ply')
        self.colmap_project = colmap.COLMAP(project_path=foldername, dense_pc='fused.ply')

        geometry = None

        if geometry is None:
            cloud = None
            try:
                cloud = self.colmap_project.get_dense()
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read PCD")
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                print("[WARNING] Failed to read points")

        if geometry is not None:
            try:
                self._scene.scene.add_geometry("__model__", geometry,
                                               rendering.MaterialRecord())
                bounds = geometry.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)

        self._apply_settings()

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Open3D GUI Example"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the radius
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum radius.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def load(self, path):
        self._scene.scene.clear_geometry()

        geometry = None
        geometry_type = o3d.io.read_file_geometry_type(path)

        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_mesh(path)
        if mesh is not None:
            if len(mesh.triangles) == 0:
                print(
                    "[WARNING] Contains 0 triangles, will read as point cloud")
                mesh = None
            else:
                mesh.compute_vertex_normals()
                if len(mesh.vertex_colors) == 0:
                    mesh.paint_uniform_color([1, 1, 1])
                geometry = mesh
            # Make sure the mesh has texture coordinates
            if not mesh.has_triangle_uvs():
                uv = np.array([[0.0, 0.0]] * (3 * len(mesh.triangles)))
                mesh.triangle_uvs = o3d.utility.Vector2dVector(uv)
        else:
            print("[Info]", path, "appears to be a point cloud")

        if geometry is None:
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                print("[WARNING] Failed to read points", path)

        if geometry is not None:
            try:
                self._scene.scene.add_geometry("__model__", geometry,
                                               self.settings.material)
                bounds = geometry.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)

    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)

    def on_mouse(self, e):
        if e.type == gui.MouseEvent.Type.BUTTON_DOWN:
            print("[debug] mouse:", (e.x, e.y))
            print(rendering.Camera.get_model_matrix(self._scene.scene.camera))

            if self.colmap_project is not None:
                l2_min = np.inf
                idx_min_dist = -1
                for idx in self.colmap_project.images.keys():
                    l2_dist = np.linalg.norm(self.colmap_project.images[idx].tvec - rendering.Camera.get_model_matrix(
                        self._scene.scene.camera)[:3, 3])

                    if l2_dist < l2_min:
                        idx_min_dist = idx

                    l2_min = min(l2_dist, l2_min)

                self._display_images.update_image(o3d.geometry.Image(self.colmap_project.images[idx_min_dist].image))
                r = self.window.content_rect
                width = 17 * self.window.theme.font_size
                self._display_images.frame = gui.Rect(r.get_right() - width * 2,
                                                      r.get_bottom() - width * 2 / 3 * 2 + 105,
                                                      width * 2,
                                                      width * 2 / 3 * 2)

        return gui.Widget.EventCallbackResult.IGNORED


def main():
    # We need to initalize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = AppWindow(1024 * 4, 768 * 4)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path)
        else:
            w.window.show_message_box("Error",
                                      "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
