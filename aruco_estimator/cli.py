
import click
import cv2

from aruco_estimator.tools.register import register
import logging
from pathlib import Path
import cv2
#TODO move to cli.py
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
        raise ValueError(f"Unsupported dictionary size: {dict_size}. Supported sizes: {list(dict_mapping.keys())}")
    
    return dict_mapping[dict_size]


@click.group()
def main():
    """ArUco Estimator CLI tool."""
    pass

@main.command('register')
@click.argument('project', type=click.Path(exists=True))
@click.option('--aruco-size', type=float, default=0.2,
              help='Size of the aruco marker in meter.')
@click.option('--dict-type', type=int, default=4,
              help='ArUco dictionary type (e.g. 4=cv2.aruco.DICT_4X4_50)')
@click.option('--show-original', is_flag=True,
              help='Show original points and cameras in visualization')
@click.option('--show', is_flag=True,
              help='Show visualization of the normalized points and cameras')
@click.option('--target-id', type=int, default=0,
              help='ID of ArUco marker to use as origin (default: 0)')
@click.option('--export-path', type=click.Path(),
              help='Path to export ArUco tag positions (default: project_path/aruco_tags.json)')
def register_cmd(project, aruco_size, dict_type, show_original, show, 
                        target_id, export_tags, export_path):
    """Normalize COLMAP poses relative to ArUco marker."""
    from aruco_estimator.sfm.colmap import COLMAPProject
    logging.basicConfig(level=logging.INFO)
    
    # Load COLMAP project using new interface
    logging.info("Loading COLMAP project...")
    
    c_project = COLMAPProject(Path(project))
    register(
        project=c_project,
        aruco_size=aruco_size,
        dict_type=get_dict_type(dict_type),
        show_original=show_original,
        show=show,
        target_id=target_id,
        export_path=export_path
    )

@main.command('align')
@click.argument('project_dirs', type=click.Path(exists=True))
@click.option('--aruco-size', type=float, default=0.2,
              help='Size of the aruco marker in meter.')
@click.option('--dict-type', type=int, default=cv2.aruco.DICT_5X5_50,
              help='ArUco dictionary type (e.g. cv2.aruco.DICT_4X4_50)')
@click.option('--show-original', is_flag=True,
              help='Show original points and cameras in visualization')
@click.option('--show', is_flag=True,
              help='Show visualization of the normalized points and cameras')
@click.option('--target-id', type=int, default=0,
              help='ID of ArUco marker to use as origin (default: 0)')
def align_cmd(project, aruco_size, dict_type, show_original, show, 
                        target_id, export_tags, export_path):
    """Normalize COLMAP poses relative to ArUco marker."""
    #TODO Merge by tag
    projects = []
    registered_projects = []
    for proj in projects:
        registered_projects.append(register(proj))
        
    align(
        projects=project,
        aruco_size=aruco_size,
        dict_type=dict_type,
        show_original=show_original,
        show=show,
        target_id=target_id,
        export_tags=export_tags,
        export_path=export_path
    )
    
    aruco_size: float = 0.2,
    dict_type: int = cv2.aruco.DICT_4X4_50, 


def show_project(p):
    pass
if __name__ == '__main__':
    main()
