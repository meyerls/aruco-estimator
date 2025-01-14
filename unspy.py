import logging

import bpy

logger = logging.getLogger(__name__)


def get_2d_coordinates(keypoint: bpy.types.Object) -> tuple[float, float]:
    """Convert keypoint world position to 2D screen coordinates.

    Args:
        keypoint: Blender object representing the keypoint
        
    Returns:
        tuple[float, float]: (x, y) screen coordinates in pixels
    """
    scene = bpy.context.scene
    render = scene.render
    camera = scene.camera

    if not camera:
        logger.error("No camera in scene")
        return (0.0, 0.0)

    # Get the camera view matrix
    view_matrix = camera.matrix_world.inverted()
    projection_matrix = camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(),
        x=render.resolution_x,
        y=render.resolution_y,
        scale_x=render.pixel_aspect_x,
        scale_y=render.pixel_aspect_y
    )

    # Get keypoint location in world space
    keypoint_location = keypoint.matrix_world.translation

    # Convert world coordinates to camera coordinates
    co_local = view_matrix @ keypoint_location

    # # Check if point is behind camera
    # if co_local.z <= 0.0:
    #     logger.debug(f"Keypoint {keypoint.name} is behind camera")
    #     return (0.0, 0.0)

    # Convert camera coordinates to NDC (Normalized Device Coordinates)
    co_4d = projection_matrix @ co_local.to_4d()

    if co_4d.w <= 0.0:
        logger.debug(f"Invalid projection for keypoint {keypoint.name}")
        return (0.0, 0.0)

    # Convert NDC to window coordinates
    ndc_x = co_4d.x / co_4d.w
    ndc_y = co_4d.y / co_4d.w

    # Convert NDC to pixel coordinates
    screen_x = (1 + ndc_x) * render.resolution_x / 2
    screen_y = (1 + ndc_y) * render.resolution_y / 2

    logger.debug(f"Keypoint {keypoint.name} screen coordinates: ({screen_x}, {screen_y})")

    return (screen_x/render.resolution_x, 1-screen_y/render.resolution_y)
