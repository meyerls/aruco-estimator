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


def get_apriltag_family(family_name: str) -> str:
    """
    Validate and return AprilTag family name.
    Args:
        family_name: AprilTag family name
    Returns:
        Validated family name
    """
    valid_families = [
        "tag16h5",
        "tag25h9",
        "tag36h10",
        "tag36h11",
        "tagCircle21h7",
        "tagCircle49h12",
        "tagCustom48h12",
        "tagStandard41h12",
        "tagStandard52h13",
    ]
    if family_name not in valid_families:
        raise ValueError(
            f"Unsupported AprilTag family: {family_name}. Supported families: {valid_families}"
        )
    return family_name


def export_marker_tags(
    project, marker_size, target_id, marker_config, transform, export_path=None
):
    """
    Export marker tag positions to JSON file.

    Args:
        project: SfM project instance
        marker_size: Size of marker in meters
        target_id: ID of target marker
        marker_config: Dictionary containing marker configuration (type, dict_type/family)
        transform: Transformation matrix used for normalization
        export_path: Path to export file (optional)
    """
    logging.info(f"Exporting {marker_config['type']} tag positions...")

    # Get all marker positions after transformation
    all_markers = {}

    if marker_config["type"] == "aruco":
        dict_type = marker_config["dict_type"]
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

    elif marker_config["type"] == "apriltag":
        # AprilTag markers are stored differently - need to access them properly
        for family_key, markers_dict in project.markers.items():
            if isinstance(family_key, str):  # AprilTag families are strings
                for marker_id, marker in markers_dict.items():
                    try:
                        all_markers[int(marker_id)] = {
                            "corners_3d": marker.corners_3d.tolist(),
                            "center_3d": marker.xyz.tolist(),
                        }
                    except Exception as e:
                        logging.warning(f"Could not export marker {marker_id}: {e}")

    export_data = {
        "marker_tags": all_markers,
        "marker_size": marker_size,
        "target_id": target_id,
        "marker_type": marker_config["type"],
        "normalization_transform": transform.tolist(),
    }

    if marker_config["type"] == "aruco":
        export_data["aruco_dict_type"] = (
            f"{marker_config['dict_type']}x{marker_config['dict_type']}"
        )
    elif marker_config["type"] == "apriltag":
        export_data["apriltag_family"] = marker_config["family"]

    if export_path is None:
        # Use project directory if available, otherwise current directory
        if hasattr(project, "project_path"):
            base_path = project.project_path
        else:
            base_path = Path.cwd()

        marker_type = marker_config["type"]
        export_path = base_path / f"{marker_type}_tags.json"

    # Save to JSON file
    with open(export_path, "w") as f:
        json.dump(export_data, f, indent=2)

    logging.info(f"{marker_config['type']} tag positions exported to {export_path}")


def visualize_project(project, original_project=None, marker_size=0.2):
    """
    Visualize the project with optional original data overlay.

    Args:
        project: Transformed SfM project instance
        original_project: Original project for comparison (optional)
        marker_size: Size of marker for coordinate frame
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
    model.add_coordinate_frame(size=marker_size)

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
    "--marker-size", type=float, default=0.2, help="Size of the marker side in meters."
)
@click.option(
    "--dict-type",
    type=int,
    default=4,
    help="ArUco dictionary type (e.g. 4=cv2.aruco.DICT_4X4_50). Only used with ArUco markers.",
)
@click.option(
    "--april",
    is_flag=True,
    help="Use AprilTag detection instead of ArUco markers",
)
@click.option(
    "--apriltag-family",
    type=str,
    default="tag36h11",
    help="AprilTag family (e.g., tag36h11, tag25h9, tag16h5). Only used with --april flag.",
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
    help="ID of marker to use as origin (default: 0)",
)
@click.option(
    "--export-path",
    type=click.Path(),
    help="Path to export marker tag positions (default: project_path/[marker_type]_tags.json)",
)
@click.option(
    "--no-export",
    is_flag=True,
    help="Skip exporting marker tag positions",
)
@click.option(
    "--no-save",
    is_flag=True,
    help="Skip saving normalized project data",
)
def register_cmd(
    project,
    marker_size,
    dict_type,
    april,
    apriltag_family,
    show_original,
    show,
    target_id,
    export_path,
    no_export,
    no_save,
):
    """Normalize COLMAP poses relative to ArUco marker or AprilTag."""
    # Load COLMAP project
    logging.info("Loading COLMAP project...")
    c_project = COLMAPProject(Path(project), sparse_folder="sparse/0")

    # Store original project state if needed for visualization
    original_project = None
    if show_original:
        original_project = deepcopy(c_project)

    # Prepare marker configuration
    if april:
        # Validate AprilTag family
        try:
            validated_family = get_apriltag_family(apriltag_family)
        except ValueError as e:
            logging.error(str(e))
            return

        marker_config = {"type": "apriltag", "family": validated_family}
        logging.info(f"Using AprilTag detection with family: {validated_family}")
    else:
        # Validate ArUco dictionary
        try:
            validated_dict_type = get_dict_type(dict_type)
        except ValueError as e:
            logging.error(str(e))
            return

        marker_config = {"type": "aruco", "dict_type": validated_dict_type}
        logging.info(f"Using ArUco detection with dictionary type: {dict_type}")

    # Perform registration (core functionality only)
    registered_project, transform, marker_results = register(
        project=c_project,
        marker_size=marker_size,
        marker_config=marker_config,
        target_id=target_id,
    )

    if registered_project is None:
        logging.error("Registration failed!")
        return
    # Handle saving
    if not no_save:
        save_normalized_project(registered_project)

    # Handle visualization
    if show:
        visualize_project(registered_project, original_project, marker_size)

    # Handle export
    if not no_export:
        export_marker_tags(
            registered_project,
            marker_size,
            target_id,
            marker_config,
            transform,
            export_path,
        )

    logging.info("Registration complete!")


@main.command("align")
@click.argument("project_dirs", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "--marker-size", type=float, default=0.2, help="Size of the marker in meters."
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
    help="ID of marker to use as origin (default: 0)",
)
def align_cmd(
    project_dirs,
    marker_size,
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

    for i, proj in enumerate(projects):
        logging.info(f"Registering project {i+1}/{len(projects)}")

        # Store original if needed
        original_proj = deepcopy(proj) if show_original else None

        # Use ArUco for alignment (could be extended to support AprilTag later)
        marker_config = {"type": "aruco", "dict_type": get_dict_type(dict_type)}

        registered_proj, _, _ = register(
            project=proj,
            marker_size=marker_size,
            marker_config=marker_config,
            target_id=target_id,
        )

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
        model.add_coordinate_frame(size=marker_size)
        model.show()

    logging.info("Alignment complete!")


if __name__ == "__main__":
    main()
