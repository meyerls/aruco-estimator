
import click
import cv2

from aruco_estimator.tools.reassign_origin import reassign_origin


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

#TODO Merge by tag



if __name__ == '__main__':
    main()
