colmap automatic_reconstructor \
    --workspace_path /Users/walkenz1/Downloads/my_ws/ \
    --image_path /Users/walkenz1/Downloads/my_ws/images \
    --camera_model SIMPLE_RADIAL 
    
    
    
    
colmap model_converter \
    --input_path "$OUTPUT_DIR/sparse/0" \
    --output_path "$OUTPUT_DIR/sparse/0" \
    --output_type TXT
