import logging
from pathlib import Path

# This patches pycolmap to fix a bug in colmap_wrapper
import aruco_estimator.patch_colmap  # noqa: F401
from aruco_estimator.localizers import ArucoLocalizer
from aruco_estimator.tools.colmap_recon import generate_colmap
from aruco_estimator.tools.downloader import DOOR_DATASET, Dataset
from aruco_estimator.visualization import ArucoVisualization
from colmap_wrapper.colmap import COLMAP

DO_COLMAP_STEP = True

DATASET_DICT = DOOR_DATASET


def main():
    dataset = Dataset()

    if DO_COLMAP_STEP:
        # Download example dataset. Door dataset is roughly 200 MB
        dataset.download_dataset(**DATASET_DICT, extract_all=False)
        # Build the colmap reconstruction just from the images
        generate_colmap(image_path=Path(dataset.dataset_path) / "images")
    else:
        dataset.download_dataset(**DATASET_DICT, extract_all=True)

    # Load Colmap project folder
    project = COLMAP(project_path=dataset.dataset_path, image_resize=0.4)

    # Init & run pose estimation of corners in 3D & estimate mean L2 distance between the four aruco corners
    aruco_localizer = ArucoLocalizer(
        photogrammetry_software=project,
        aruco_size=dataset.scale,
        target_id=DATASET_DICT["tag_id"],
        dict_type=DATASET_DICT["dict_type"],
    )
    aruco_distance, aruco_corners_3d = aruco_localizer.run()
    logging.info("Size of the unscaled aruco markers: ", aruco_distance)

    # Calculate scaling factor, apply it to the scene and save scaled point cloud
    dense, scale_factor = aruco_localizer.apply()
    logging.info("Point cloud and poses are scaled by: ", scale_factor)
    logging.info(
        "Size of the scaled (true to scale) aruco markers in meters: ",
        aruco_distance * scale_factor,
    )

    # Visualization of the scene and rays
    vis = ArucoVisualization(aruco_colmap=aruco_localizer)
    vis.visualization(frustum_scale=0.7, point_size=0.1)

    # Write Data
    aruco_localizer.write_data()


if __name__ == "__main__":
    main()
