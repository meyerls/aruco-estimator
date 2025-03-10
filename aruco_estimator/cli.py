
import click
import cv2

from aruco_estimator.tools.reassign_origin import reassign_origin
from aruco_estimator.tools.reverse_project import reverse_project


@click.group()
def main():
    """ArUco Estimator CLI tool."""
    pass


@main.command('reassign-origin')
@click.argument('colmap_project', type=click.Path(exists=True))
@click.option('--aruco-size', type=float, default=0.2,
              help='Size of the aruco marker in meter.')
@click.option('--dict-type', type=int, default=cv2.aruco.DICT_4X4_50,
              help='ArUco dictionary type (e.g. cv2.aruco.DICT_4X4_50)')
@click.option('--show-original', is_flag=True,
              help='Show original points and cameras in visualization')
@click.option('--visualize', is_flag=True,
              help='Show visualization of the normalized points and cameras')
@click.option('--target-id', type=int, default=0,
              help='ID of ArUco marker to use as origin (default: 0)')
@click.option('--export-tags', is_flag=True,
              help='Export ArUco tag positions to a JSON file')
@click.option('--export-path', type=click.Path(),
              help='Path to export ArUco tag positions (default: project_path/aruco_tags.json)')
def reassign_origin_cmd(colmap_project, aruco_size, dict_type, show_original, visualize, 
                        target_id, export_tags, export_path):
    """Normalize COLMAP poses relative to ArUco marker."""
    reassign_origin(
        colmap_project=colmap_project,
        aruco_size=aruco_size,
        dict_type=dict_type,
        show_original=show_original,
        visualize=visualize,
        target_id=target_id,
        export_tags=export_tags,
        export_path=export_path
    )


@main.command('reverse-project')
@click.argument('colmap_path', type=click.Path(exists=True))
@click.argument('images_path', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--key-positions', type=click.Path(exists=True),
              help='Path to key positions text file')
@click.option('--grid-min', type=(float, float, float), default=(-2.0, -2.0, -2.0),
              help='Minimum coordinates for grid (x,y,z)')
@click.option('--grid-max', type=(float, float, float), default=(2.0, 2.0, 2.0),
              help='Maximum coordinates for grid (x,y,z)')
@click.option('--grid-points', type=int, default=4,
              help='Number of points per dimension in grid')
@click.option('--skip-copy', is_flag=True,
              help='Skip saving visualization images, only save labels')
def reverse_project_cmd(colmap_path, images_path, output_dir, key_positions, grid_min, grid_max, grid_points, skip_copy):
    """Project 3D points onto images and create dataset."""
    reverse_project(
        colmap_path=colmap_path,
        images_path=images_path,
        output_dir=output_dir,
        key_positions_path=key_positions,
        grid_min=grid_min,
        grid_max=grid_max,
        grid_points=grid_points,
        skip_copy=skip_copy
    )


if __name__ == '__main__':
    main()
