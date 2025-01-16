# First, set the base directory as a variable
BASE_DIR="/Users/walkenz1/Projects/aruco-estimator/data/1_15_25/C2"

# Then use the variable in the COLMAP command
colmap automatic_reconstructor \
    --workspace_path "${BASE_DIR}/" \
    --image_path "${BASE_DIR}/images" \
    --single_camera 1
mv ${BASE_DIR}/sparse/0/* "${BASE_DIR}/sparse/"
python aruco_estimator/tools/reassign_origin.py \
    --colmap_project "${BASE_DIR}"

rm -rf "${BASE_DIR}/sparse"

mv "${BASE_DIR}/normalized" "${BASE_DIR}/colmap"  