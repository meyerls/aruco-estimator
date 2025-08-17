from aruco_estimator.cli import visualize_project
from aruco_estimator.sfm.colmap import COLMAPProject
from aruco_estimator.utils import (
    get_corners_at_origin,
    get_transformation_between_clouds,
)
from aruco_estimator.tools.merge import merge_projects, align_projects
import cv2

# Load both projects
project_A = COLMAPProject(
    ".data/_alley/",
    sparse_folder="sparse/0",
)
project_B = COLMAPProject(".data/_park", sparse_folder="sparse/0")

# Parameters
target_id = 0  # ArUco marker ID to use for alignment
aruco_size = 0.15

print("Detecting ArUco markers...")
# Detect markers in both projects
aruco_resultsA = project_A.detect_markers(dict_type=cv2.aruco.DICT_4X4_50)
aruco_resultsB = project_B.detect_markers(dict_type=cv2.aruco.DICT_4X4_50)

print(f"Project A found {len(aruco_resultsA)} markers")
print(f"Project B found {len(aruco_resultsB)} markers")

# Check if target marker exists in both projects
if target_id not in aruco_resultsA:
    raise ValueError(f"Target marker {target_id} not found in project A")
if target_id not in aruco_resultsB:
    raise ValueError(f"Target marker {target_id} not found in project B")

print(f"Aligning projects using marker {target_id}...")
# Calculate transformation to align project B to project A coordinate system
transform = get_transformation_between_clouds(
    aruco_resultsB[target_id], aruco_resultsA[target_id]
)

# Apply transformation to project B
print("Applying transformation to project B...")
project_B.transform(transform)

# Alternative: Use the align_projects function for automatic alignment
# aligned_projects = align_projects([project_A, project_B], target_id=0)
# project_A, project_B = aligned_projects

print("Merging projects with image copying...")
# Option 1: Merge with image copying (creates fresh COLMAP project)
merged_project = merge_projects(
    [project_A, project_B],
    output_path="./merged_output/sparse/0/",
    images_path="./merged_output/images/",
)

# Save the merged project
print("Saving merged project...")
merged_project.save("./merged_output/sparse/0/")

print("Visualizing merged project...")
visualize_project(merged_project)

print("Merge complete!")
print(f"- Total cameras: {len(merged_project.cameras)}")
print(f"- Total images: {len(merged_project.images)}")
print(f"- Total 3D points: {len(merged_project.points3D)}")
print(
    f"- Total markers: {sum(len(markers) for markers in merged_project.markers.values())}"
)

# Optional: Alternative workflow without image copying
print("\n" + "=" * 50)
print("Alternative: Merge without copying images...")

merged_project_no_copy = merge_projects(
    [project_A, project_B],
    output_path="./merged_output_no_copy/sparse/0/",
    images_path=None,  # No image copying
)

merged_project_no_copy.save("./merged_output_no_copy/sparse/0/")
print("Lightweight merge complete (no image copying)")

# Optional: Normalize to specific ArUco marker at origin
print("\n" + "=" * 50)
print("Optional: Normalizing to place marker at origin...")

# Get 3D corners for normalization from merged project
target_corners_3d = merged_project.markers[cv2.aruco.DICT_4X4_50][target_id].corners_3d

# Calculate transform to place marker at origin with correct scale
normalize_transform = get_transformation_between_clouds(
    target_corners_3d, get_corners_at_origin(side_length=aruco_size)
)

# Apply normalization
merged_project.transform(normalize_transform)
merged_project.save("./normalized_output/sparse/0/")

print("Normalization complete - marker now at origin with correct scale")
print(f"Marker corners at origin: {get_corners_at_origin(side_length=aruco_size)}")
print(
    f"Transformed marker corners: {merged_project.markers[cv2.aruco.DICT_4X4_50][target_id].corners_3d}"
)
