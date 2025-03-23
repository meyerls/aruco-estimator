import glob
import os
import shutil
import subprocess
from pathlib import Path

import pycolmap
from PIL import Image  # For image resizing

from .downloader import extract_from_zip


def resize_images(image_path, max_size=2048):
    """
    Resize all images in the given directory to have a maximum dimension of max_size.
    Creates a backup of original images first.
    """
    # Create backup directory
    parent_path = Path(image_path).parent
    backup_path = parent_path / "original_images"
    os.makedirs(backup_path, exist_ok=True)

    # Get all image files
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_files.extend(glob.glob(os.path.join(image_path, ext)))

    for img_file in image_files:
        # Backup original image
        filename = os.path.basename(img_file)
        shutil.copy2(img_file, os.path.join(backup_path, filename))

        # Resize image
        with Image.open(img_file) as img:
            width, height = img.size

            # Calculate new dimensions
            if width > height:
                if width > max_size:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
            else:
                if height > max_size:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                else:
                    new_width, new_height = width, height

            # Only resize if needed
            if new_width != width or new_height != height:
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                resized_img.save(img_file, quality=95)
                print(
                    f"Resized {filename} from {width}x{height} to {new_width}x{new_height}"
                )


DO_DENSE_RECONSTRUCTION = True


def generate_colmap(image_path: str):
    """
    # Given image_path /images, invoke pycolmap to generate 4 files:
    #    /sparse/cameras.bin
    #    /sparse/images.bin
    #    /sparse/points3D.bin
    #    /fused.ply
    """

    parent_path = Path(image_path).parent

    # Define paths
    database_path = parent_path / "colmap.db"
    sparse_output = parent_path / "sparse"
    dense_workspace = parent_path / "dense"
    fused_output = parent_path / "fused.ply"

    # Delete everything in the superfolder for images
    # because we're about to re-do the colmap reconstruction
    if sparse_output.is_dir():
        shutil.rmtree(sparse_output)

    if dense_workspace.is_dir():
        shutil.rmtree(dense_workspace)

    if database_path.is_file():
        os.remove(database_path)

    if fused_output.is_file():
        os.remove(fused_output)

    # Create necessary directories if they don't exist
    os.makedirs(sparse_output, exist_ok=True)
    os.makedirs(dense_workspace, exist_ok=True)

    resize_images(image_path, max_size=2048)

    # 1. Extract features and create the COLMAP database
    sift_options = pycolmap.SiftExtractionOptions()
    sift_options.num_threads = 4
    sift_options.max_image_size = 1024

    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_path,
        camera_model="PINHOLE",  # or 'SIMPLE_RADIAL', etc.
        camera_mode=pycolmap.CameraMode.SINGLE,  # instead of single_camera=True
        reader_options=pycolmap.ImageReaderOptions(
            default_focal_length_factor=1.0  # keep other relevant settings
        ),
        sift_options=sift_options,
    )

    # 2. Match features across images
    pycolmap.match_exhaustive(
        database_path=database_path,
        # If you want to pass SiftMatchingOptions:
        sift_options=pycolmap.SiftMatchingOptions(
            # other matching params here...
        ),
        # device=pycolmap.Device.cpu,
    )

    # 3. Run incremental mapping (sparse reconstruction)
    # This creates a subfolder (usually named "0") under /sparse containing:
    #   - cameras.bin
    #   - images.bin
    #   - points3D.bin
    sparse_models_dict = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_path,
        output_path=sparse_output,
        # mapper_options=mapper_options,
    )

    # Move the files from the subfolder (e.g., /sparse/0) to /sparse directly:
    model_subfolder = os.path.join(sparse_output, "0")
    for file_name in ["cameras.bin", "images.bin", "points3D.bin"]:
        src_file = os.path.join(model_subfolder, file_name)
        dst_file = os.path.join(sparse_output, file_name)
        if os.path.exists(src_file):
            os.replace(src_file, dst_file)

    # 4. Prepare for dense reconstruction by undistorting images using the sparse model

    if False:
        options = pycolmap.UndistortCameraOptions()
        # (we will need to downsample the images to avoid a RAM overload during the
        # patch_match_stereo step below)
        # Downscale images so that their largest dimension is 1024
        options.max_image_size = 2048

        # Important: These additional parameters ensure consistency
        options.blank_pixels = True
        options.min_scale = 0.2  # Allow more scaling
        options.max_scale = 2.0
        options.roi_min_x = 0
        options.roi_min_y = 0
        options.roi_max_x = 1.0
        options.roi_max_y = 1.0

        # Add this line to ensure consistency between the image and camera models
        # options.image_height = options.max_image_size
        # options.image_width = options.max_image_size
        pycolmap.undistort_images(
            output_path=dense_workspace,
            input_path=sparse_output,
            image_path=image_path,
            output_type="COLMAP",
            undistort_options=options,
        )
    else:
        # Run the image undistorter with max_image_size (e.g., 2048)
        cmd_undistort = [
            "colmap",
            "image_undistorter",
            "--image_path",
            str(image_path),
            "--input_path",
            str(sparse_output),
            "--output_path",
            str(dense_workspace),
            "--output_type",
            "COLMAP",
            "--max_image_size",
            "2048",
        ]

        subprocess.run(cmd_undistort, check=True)

    if DO_DENSE_RECONSTRUCTION:
        # 5. Compute depth maps with patch-match stereo
        # (requires compilation with CUDA)
        # pycolmap doesn't appear to have this command, so we'll use subprocess
        # pycolmap.patch_match_stereo(workspace_path=dense_workspace)
        command = [
            "colmap",
            "patch_match_stereo",
            "--workspace_path",
            str(dense_workspace),
        ]

        subprocess.run(command, check=True)

        # 6. Fuse depth maps into a dense point cloud saved as fused.ply
        pycolmap.stereo_fusion(workspace_path=dense_workspace, output_path=fused_output)
    else:
        # KLUDGE: Just grab the file from the ZIP for now
        extract_from_zip(
            src_path="door/fused.ply",
            dst_path=fused_output,
            zip_path=str(parent_path) + ".zip",
        )
