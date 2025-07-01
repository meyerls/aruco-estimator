import click
import cv2
import json
import logging
from copy import deepcopy
from pathlib import Path

from aruco_estimator.tools.register import register
from aruco_estimator.tools.merge import align_projects
from aruco_estimator.sfm.colmap import COLMAPProject
from aruco_estimator.visualization import VisualizationModel

logging.basicConfig(level=logging.INFO)


def get_dict_type(dict_size: int) -> int:
    """
    Map dictionary size number to OpenCV ArUco dictionary constant.
    Args:
        dict_size: Dictionary size (4, 5, 6, 7)
    Returns:
        OpenCV ArUco dictionary constant
    """
    dict_mapping = {
        4: cv2.aruco.DICT_4X4_50,
        5: cv2.aruco.DICT_5X5_50,
        6: cv2.aruco.DICT_6X6_50,
        7: cv2.aruco.DICT_7X7_50,
    }
    if dict_size not in dict_mapping:
        raise ValueError(
            f"Unsupported dictionary size: {dict_size}. Supported sizes: {list(dict_mapping.keys())}"
        )
    return dict_mapping[dict_size]


def export_aruco_tags(
    project, aruco_size, target_id, dict_type, transform, export_path=None
):
    """
    Export ArUco tag positions to JSON file.

    Args:
        project: SfM project instance
        aruco_size: Size of ArUco marker in meters
        target_id: ID of target ArUco marker
        dict_type: ArUco dictionary type
        transform: Transformation matrix used for normalization
        export_path: Path to export file (optional)
    """
    logging.info("Exporting ArUco tag positions...")

    # Get all marker positions after transformation
    all_markers = {}
    for dict_type_key, markers_dict in project.markers.items():
        if dict_type_key == dict_type:  # Only export markers from the dict we used
            for marker_id, marker in markers_dict.items():
                try:
                    # Get transformed 3D corners from the project
                    transformed_corners = project.get(dict_type_key, {}).get(
                        marker_id, None
                    )

                    all_markers[int(marker_id)] = {
                        "corners_3d": transformed_corners.tolist(),
                        "center_3d": transformed_corners.mean(axis=0).tolist(),
                    }
                except Exception as e:
                    logging.warning(f"Could not export marker {marker_id}: {e}")

    export_data = {
        "aruco_tags": all_markers,
        "aruco_size": aruco_size,
        "target_id": target_id,
        "dict_type": f"{dict_type}x{dict_type}",
        "normalization_transform": transform.tolist(),
    }

    if export_path is None:
        # Use project directory if available, otherwise current directory
        if hasattr(project, "project_path"):
            base_path = project.project_path
        else:
            base_path = Path.cwd()
        export_path = base_path / "aruco_tags.json"

    # Save to JSON file
    with open(export_path, "w") as f:
        json.dump(export_data, f, indent=2)

    logging.info(f"ArUco tag positions exported to {export_path}")


def visualize_project(project, original_project=None, aruco_size=0.2):
    """
    Visualize the project with optional original data overlay.

    Args:
        project: Transformed SfM project instance
        original_project: Original project for comparison (optional)
        aruco_size: Size of ArUco marker for coordinate frame
    """
    model = VisualizationModel()
    model.create_window()

    # Add original data in gray if provided (show first so it's in background)
    if original_project:
        model.add_project(
            original_project,
            points_config={"color": [0.7, 0.7, 0.7]},
            cameras_config={
                "scale": 0.25,
                "color": [0.7, 0.7, 0.7],
                "show_images": False,
            },
            markers_config={
                "show_detection_lines": False,
                "corner_size": 0.03,
            },
        )

    # Add transformed data (foreground)
    model.add_project(
        project,
        points_config={},  # Use default colors from point cloud
        cameras_config={"scale": 0.25, "color": [1, 0, 0]},  # Red cameras
        markers_config={
            "show_detection_lines": True,
            "detection_line_color": [0, 1, 0],  # Green detection lines
            "corner_size": 0.05,
        },
    )

    # Add coordinate frame at origin
    model.add_coordinate_frame(size=aruco_size)

    # Show visualization
    model.show()


def save_normalized_project(project):
    """
    Save the normalized project data.

    Args:
        project: SfM project instance to save
    """
    logging.info("Saving normalized data...")
    output_dir = Path("normalized") / "sparse"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save transformed data using the project's save method
    project.save(str(output_dir))
    logging.info(f"Normalized data saved to {output_dir}")


@click.group()
def main():
    """ArUco Estimator CLI tool."""
    pass


@main.command("register")
@click.argument("project", type=click.Path(exists=True))
@click.option(
    "--aruco-size", type=float, default=0.2, help="Size of the aruco marker in meter."
)
@click.option(
    "--dict-type",
    type=int,
    default=4,
    help="ArUco dictionary type (e.g. 4=cv2.aruco.DICT_4X4_50)",
)
@click.option(
    "--show-original",
    is_flag=True,
    help="Show original points and cameras in visualization",
)
@click.option(
    "--show",
    is_flag=True,
    help="Show visualization of the normalized points and cameras",
)
@click.option(
    "--target-id",
    type=int,
    default=0,
    help="ID of ArUco marker to use as origin (default: 0)",
)
@click.option(
    "--export-path",
    type=click.Path(),
    help="Path to export ArUco tag positions (default: project_path/aruco_tags.json)",
)
@click.option(
    "--no-export",
    is_flag=True,
    help="Skip exporting ArUco tag positions",
)
@click.option(
    "--no-save",
    is_flag=True,
    help="Skip saving normalized project data",
)
def register_cmd(
    project,
    aruco_size,
    dict_type,
    show_original,
    show,
    target_id,
    export_path,
    no_export,
    no_save,
):
    """Normalize COLMAP poses relative to ArUco marker."""
    # Load COLMAP project
    logging.info("Loading COLMAP project...")
    c_project = COLMAPProject(Path(project))

    # Store original project state if needed for visualization
    original_project = None
    if show_original:
        original_project = deepcopy(c_project)

    # Perform registration (core functionality only)
    registered_project, transform, aruco_results = register(
        project=c_project,
        aruco_size=aruco_size,
        dict_type=get_dict_type(dict_type),
        target_id=target_id,
    )

    if registered_project is None:
        logging.error("Registration failed!")
        return

    # Handle visualization
    if show:
        visualize_project(registered_project, original_project, aruco_size)

    # Handle export
    if not no_export:
        export_aruco_tags(
            registered_project,
            aruco_size,
            target_id,
            get_dict_type(dict_type),
            transform,
            export_path,
        )

    # Handle saving
    if not no_save:
        save_normalized_project(registered_project)

    logging.info("Registration complete!")


@main.command("align")
@click.argument("project_dirs", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "--aruco-size", type=float, default=0.2, help="Size of the aruco marker in meter."
)
@click.option(
    "--dict-type",
    type=int,
    default=5,
    help="ArUco dictionary type (4, 5, 6, 7)",
)
@click.option(
    "--show-original",
    is_flag=True,
    help="Show original points and cameras in visualization",
)
@click.option(
    "--show",
    is_flag=True,
    help="Show visualization of the aligned projects",
)
@click.option(
    "--target-id",
    type=int,
    default=0,
    help="ID of ArUco marker to use as origin (default: 0)",
)
def align_cmd(
    project_dirs,
    aruco_size,
    dict_type,
    show_original,
    show,
    target_id,
):
    """Align multiple COLMAP projects using ArUco markers."""
    # Load all projects
    projects = []
    for proj_dir in project_dirs:
        logging.info(f"Loading project: {proj_dir}")
        projects.append(COLMAPProject(Path(proj_dir)))

    # Register each project individually first
    registered_projects = []
    # transforms = []
    target_corners_3d = aruco_results[target_id]
    logging.info(f"Using marker {target_id} for normalization")
    logging.debug(f"Target corners 3D: {target_corners_3d}")

    # Calculate normalization transform with scaling
    transform = get_transformation_between_clouds(
        target_corners_3d, get_corners_at_origin(side_length=aruco_size)
    )

    # Apply normalizatio
    for i, proj in enumerate(projects):
        logging.info(f"Registering project {i+1}/{len(projects)}")

        # Store original if needed
        original_proj = deepcopy(proj) if show_original else None

        registered_proj = proj.detect_markers()

        if registered_proj is not None:
            registered_projects.append(registered_proj)
        else:
            logging.warning(f"Failed to register project {proj_dir}")

    if not registered_projects:
        logging.error("No projects could be registered!")
        return

    # Align projects (this function needs to be implemented)
    aligned_projects = align_projects(
        projects=registered_projects,
        # aruco_size=aruco_size,
        # dict_type=get_dict_type(dict_type),
        target_id=target_id,
    )

    # Handle visualization
    if show:
        # Visualize all aligned projects together
        model = VisualizationModel()
        model.create_window()

        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

        for i, proj in enumerate(aligned_projects):
            color = colors[i % len(colors)]
            model.add_project(
                proj,
                points_config={"color": color},
                cameras_config={"scale": 0.25, "color": color},
                markers_config={
                    "show_detection_lines": True,
                    "detection_line_color": color,
                    "corner_size": 0.05,
                },
            )

        # Add coordinate frame at origin
        model.add_coordinate_frame(size=aruco_size)
        model.show()

    logging.info("Alignment complete!")


if __name__ == "__main__":
    main()
