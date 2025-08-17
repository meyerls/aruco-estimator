#!/usr/bin/env python3
"""
Convert COLMAP sparse point cloud to PLY format.

This script reads COLMAP sparse reconstruction data (cameras.bin/txt, images.bin/txt, points3D.bin/txt)
and converts the 3D points to a PLY point cloud file.

Usage:
    python colmap_sparse_to_ply.py <colmap_sparse_path> [output_ply_path]

Example:
    python colmap_sparse_to_ply.py path/to/sparse/0 output.ply
    python colmap_sparse_to_ply.py path/to/sparse/0  # outputs to sparse_points.ply
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Import from your COLMAP module - adjust import path as needed
from aruco_estimator.sfm.colmap import read_model, COLMAPProject


def write_ply_header(file, num_points):
    """Write PLY file header."""
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    file.write(header)


def write_ply_binary_header(file, num_points):
    """Write binary PLY file header."""
    header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    file.write(header.encode("ascii"))


def sparse_to_ply_simple(sparse_path, output_path, binary=False):
    """
    Convert COLMAP sparse reconstruction to PLY using direct file reading.

    Args:
        sparse_path: Path to COLMAP sparse folder (containing cameras, images, points3D files)
        output_path: Output PLY file path
        binary: If True, write binary PLY format
    """
    print(f"Reading COLMAP model from: {sparse_path}")

    # Read COLMAP model
    try:
        cameras, images, points3D = read_model(sparse_path)
    except Exception as e:
        print(f"Error reading COLMAP model: {e}")
        return False

    print(f"Loaded {len(points3D)} 3D points")

    if len(points3D) == 0:
        print("No 3D points found in the sparse reconstruction")
        return False

    print(f"Writing PLY file to: {output_path}")

    if binary:
        # Write binary PLY
        import struct

        with open(output_path, "wb") as f:
            write_ply_binary_header(f, len(points3D))

            for point_id, point in points3D.items():
                # Write XYZ as floats
                f.write(
                    struct.pack(
                        "<fff",
                        float(point.xyz[0]),
                        float(point.xyz[1]),
                        float(point.xyz[2]),
                    )
                )
                # Write RGB as unsigned chars
                f.write(
                    struct.pack(
                        "<BBB", int(point.rgb[0]), int(point.rgb[1]), int(point.rgb[2])
                    )
                )
    else:
        # Write ASCII PLY
        with open(output_path, "w") as f:
            write_ply_header(f, len(points3D))

            for point_id, point in points3D.items():
                x, y, z = point.xyz
                r, g, b = point.rgb
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    print(f"Successfully converted {len(points3D)} points to PLY format")
    return True


def sparse_to_ply_with_project(
    project_path, output_path, sparse_folder="sparse/0", binary=False
):
    """
    Convert COLMAP sparse reconstruction to PLY using COLMAPProject class.

    Args:
        project_path: Path to COLMAP project directory
        output_path: Output PLY file path
        sparse_folder: Relative path to sparse folder
        binary: If True, write binary PLY format
    """
    print(f"Loading COLMAP project from: {project_path}")

    try:
        # Create COLMAP project instance
        project = COLMAPProject(project_path, sparse_folder=sparse_folder)

        # Access the loaded points3D data
        points3D = project._points3D

        print(f"Loaded {len(points3D)} 3D points from project")

        if len(points3D) == 0:
            print("No 3D points found in the sparse reconstruction")
            return False

        print(f"Writing PLY file to: {output_path}")

        if binary:
            # Write binary PLY
            import struct

            with open(output_path, "wb") as f:
                write_ply_binary_header(f, len(points3D))

                for point_id, point in points3D.items():
                    # Write XYZ as floats
                    f.write(
                        struct.pack(
                            "<fff",
                            float(point.xyz[0]),
                            float(point.xyz[1]),
                            float(point.xyz[2]),
                        )
                    )
                    # Write RGB as unsigned chars
                    f.write(
                        struct.pack(
                            "<BBB",
                            int(point.rgb[0]),
                            int(point.rgb[1]),
                            int(point.rgb[2]),
                        )
                    )
        else:
            # Write ASCII PLY
            with open(output_path, "w") as f:
                write_ply_header(f, len(points3D))

                for point_id, point in points3D.items():
                    x, y, z = point.xyz
                    r, g, b = point.rgb
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

        print(f"Successfully converted {len(points3D)} points to PLY format")
        return True

    except Exception as e:
        print(f"Error processing COLMAP project: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert COLMAP sparse point cloud to PLY format"
    )
    parser.add_argument(
        "input_path", help="Path to COLMAP sparse folder or project directory"
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default="sparse_points.ply",
        help="Output PLY file path (default: sparse_points.ply)",
    )
    parser.add_argument("--binary", action="store_true", help="Write binary PLY format")
    parser.add_argument(
        "--project-mode",
        action="store_true",
        help="Use COLMAPProject class (input should be project directory)",
    )
    parser.add_argument(
        "--sparse-folder",
        default="sparse/0",
        help="Sparse folder path relative to project (default: sparse/0)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = False

    if args.project_mode:
        # Use COLMAPProject class
        success = sparse_to_ply_with_project(
            str(input_path), str(output_path), args.sparse_folder, args.binary
        )
    else:
        # Direct sparse folder reading
        success = sparse_to_ply_simple(str(input_path), str(output_path), args.binary)

    if success:
        print(f"\n✓ Conversion completed successfully!")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
    else:
        print(f"\n✗ Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
