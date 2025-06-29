ffmpeg -i park.mp4 -vf "select='not(mod(n,5))'" -vsync vfr images/frame_%04d.png



# # Then use the variable in the COLMAP command
colmap automatic_reconstructor \
    --workspace_path "_park/" \
    --image_path "_park/images" \
    --dense 0 