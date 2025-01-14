#!/bin/bash

# Input parameters
IMAGE_DIR="$1"
OUTPUT_DIR="$2"
QUALITY="${3:-high}"  # Optional: defaults to high
GPU_INDEX="${4:-0}"   # Optional: defaults to 0

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

colmap automatic_reconstructor \
    --workspace_path /Users/walkenz1/Downloads/my_ws/ \
    --image_path /Users/walkenz1/Downloads/my_ws/images \
    --camera_model SIMPLE_RADIAL 
    
    
    
    

# Check if reconstruction was successful
if [ ! -d "$OUTPUT_DIR/sparse/0" ]; then
    echo "Reconstruction failed!"
    exit 1
fi

# Step 2: Convert to text format
echo "Converting to text format..."
colmap model_converter \
    --input_path "$OUTPUT_DIR/sparse/0" \
    --output_path "$OUTPUT_DIR/sparse/0" \
    --output_type TXT

# Check if conversion was successful
if [ ! -f "$OUTPUT_DIR/sparse/0/cameras.txt" ]; then
    echo "Conversion to text format failed!"
    exit 1
fi

echo "Process completed successfully!"
echo "Output files can be found in: $OUTPUT_DIR/sparse/0/"
echo "Generated files:"
echo "- cameras.txt"
echo "- images.txt"
echo "- points3D.txt"