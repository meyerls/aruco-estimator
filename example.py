from aruco_estimator.aruco_scale_factor import ArucoScaleFactor
from aruco_estimator.download import Dataset
from aruco_estimator.visualization import ArucoVisualization
from colmap_wrapper.dataloader import COLMAPLoader

# Download example dataset. Door dataset is roughly 200 MB
dataset = Dataset()
dataset.download_door_dataset()

# Load Colmap project folder

# project = COLMAPProject(project_path=dataset.dataset_path)  # , image_resize=0.4)

project = COLMAPLoader(project_path=dataset.dataset_path)

# import code

# code.interact(local=locals())

# Init & run pose estimation of corners in 3D & estimate mean L2 distance between the four aruco corners
aruco_scale_factor = ArucoScaleFactor(
    photogrammetry_software=project, aruco_size=dataset.scale
)
aruco_distance, aruco_corners_3d = aruco_scale_factor.run()
print("Size of the unscaled aruco markers: ", aruco_distance)

# Calculate scaling factor, apply it to the scene and save scaled point cloud
dense, scale_factor = aruco_scale_factor.apply()
print("Point cloud and poses are scaled by: ", scale_factor)
print(
    "Size of the scaled (true to scale) aruco markers in meters: ",
    aruco_distance * scale_factor,
)

# Visualization of the scene and rays
vis = ArucoVisualization(aruco_colmap=aruco_scale_factor)
vis.visualization(frustum_scale=0.7, point_size=0.1)

# Write Data
aruco_scale_factor.write_data()
