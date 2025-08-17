#!/usr/bin/env python3
"""
Convert JSON marker data to CSV format.
One row per marker with all corner coordinates and metadata.

Usage:
    python json_markers_to_csv.py <input_json> [output_csv]

Example:
    python json_markers_to_csv.py markers.json markers.csv
    python json_markers_to_csv.py markers.json  # outputs to markers_output.csv
"""

import json
import csv
import argparse
import sys
from pathlib import Path


def convert_markers_to_csv(json_data, output_path):
    """
    Convert marker JSON data to CSV format.
    One row per marker with all corner coordinates.
    """

    marker_tags = json_data.get("marker_tags", {})
    marker_type = json_data.get("marker_type", "")
    apriltag_family = json_data.get("apriltag_family", "")

    with open(output_path, "w", newline="") as csvfile:
        fieldnames = [
            "marker_id",
            "tl_x",
            "tl_y",
            "tl_z",
            "tr_x",
            "tr_y",
            "tr_z",
            "br_x",
            "br_y",
            "br_z",
            "bl_x",
            "bl_y",
            "bl_z",
            "marker_type",
            "apriltag_family",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for marker_id, marker_data in marker_tags.items():
            corners_3d = marker_data.get("corners_3d", [])

            row = {
                "marker_id": marker_id,
                "marker_type": marker_type,
                "apriltag_family": apriltag_family,
            }

            # Add corner data (tl, tr, br, bl)
            corner_names = ["tl", "tr", "br", "bl"]
            for i, corner_name in enumerate(corner_names):
                if i < len(corners_3d):
                    row[f"{corner_name}_x"] = corners_3d[i][0]
                    row[f"{corner_name}_y"] = corners_3d[i][1]
                    row[f"{corner_name}_z"] = corners_3d[i][2]
                else:
                    row[f"{corner_name}_x"] = 0
                    row[f"{corner_name}_y"] = 0
                    row[f"{corner_name}_z"] = 0

            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON marker data to CSV format"
    )
    parser.add_argument("input_json", help="Input JSON file path")
    parser.add_argument("output_csv", nargs="?", help="Output CSV file path")

    args = parser.parse_args()

    input_path = Path(args.input_json)

    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        sys.exit(1)

    # Generate output path if not provided
    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        output_path = input_path.with_name(input_path.stem + "_output.csv")

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Read JSON data
        with open(input_path, "r") as f:
            json_data = json.load(f)

        # Convert to CSV
        convert_markers_to_csv(json_data, output_path)

        marker_count = len(json_data.get("marker_tags", {}))
        print(f"Converted {marker_count} markers to CSV: {output_path}")

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
