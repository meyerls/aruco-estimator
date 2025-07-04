<p align="center" width="100%">
    <img width="100%" src="assets/wood.png">
</p>

# Automatic Registration Estimation Based on ArUco Markers

<a href="https://pypi.org/project/aruco-estimator/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/aruco-estimator"></a>
<a href="https://pypi.org/project/aruco-estimator/"><img alt="PyPI" src="https://img.shields.io/pypi/v/aruco-estimator"></a>
<a href="https://github.com/meyerls/aruco-estimator/actions"><img alt="GitHub Workflow Status" src="https://img.shields.io/github/workflow/status/meyerls/aruco-estimator/Python%20package"></a>
<a href="https://github.com/meyerls/aruco-estimator/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/github/license/meyerls/aruco-estimator"></a>

## About

This project automatically computes the correct scale and registration of 3D reconstructions generated with [COLMAP](https://colmap.github.io/) by placing one or more ArUco markers in the scene. This module does not perform the reconstruction task itself so a colmap installation isn't needed.
## Installation

This repository is tested on Python 3.6+ and can be installed from PyPI:

```bash
pip install aruco-estimator
```
or 
```
pip install git+https://github.com/meyerls/aruco-estimator
```

## Usage

### Test Data

Download and extract the test dataset:

```bash
wget -O door.zip "https://faubox.rrze.uni-erlangen.de/dl/fiUNWMmsaEAavXHfjqxfyXU9/door.zip"
unzip door.zip
```

### Registration

Test the registration functionality with the example project:

```bash
aruco-estimator register ./door --target-id 7 --dict-type 4 --show --aruco-size 0.15

```

<p align="center" width="100%">
    <img width="100%" src="assets/door.png?raw=true">
</p>


### Merging 



### Scripting

Test the registration functionality with the example project:

```bash
aruco-estimator align .data/_park .data/_alley--target-id 7 --dict-type 4 --show --aruco-size 0.15

```

``` python 
from aruco_estimator.sfm.colmap import COLMAPProject
from aruco_estimator.utils import get_transformation_between_clouds,get_corners_at_origin
import cv2
project = COLMAPProject('./door', )
/sparse/0
target_id = 7
aruco_size = .15
aruco_results = project.detect_markers(dict_type=cv2.aruco.DICT_4X4_50)

# Get 3D corners for normalization
target_corners_3d = aruco_results[target_id]
print(target_corners_3d) 

# Calculate 4x4 transform with scaling so tag is at the origin 
transform = get_transformation_between_clouds(target_corners_3d, get_corners_at_origin(side_length=aruco_size))

# Apply normalization to the project
print("Normalizing poses and 3D points...")
project.transform(transform)
project.save("./transformed_output/")

```

<p align="center" width="100%">
    <img width="100%" src="assets/output.gif?raw=true">
</p>

### Dense Clouds
Current Script expects {project_dir}/fused.ply for dense cloud alignment and viz

## Known Limitations 

- Only SIMPLE_RADIAL and PINHOLE camera models are supported
- Aruco boards are not uniquely supported 
- Pose estimation is not robust to false detections; ransac would be beneficial
- Only COLMAP .bin and .txt models are supported

## Roadmap
- [ ] Update README with multi tag examples
- [ ] Improved pose estimation robustness
- [ ] Implement the merge by tag tool 
- [ ] Support for additional camera models
- [ ] Geo-referencing of ArUco markers with Earth coordinate system using GPS or RT

## Troubleshooting

### OpenCV ArUco Module Issues

If cv2 doesn't detect the ArUco marker module, try reinstalling OpenCV:

```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python opencv-contrib-python
```

## Acknowledgements

- Code for reading binary COLMAP data is partly borrowed from [COLMAP Utility Scripts](https://github.com/uzh-rpg/colmap_utils) by [uzh-rpg](https://github.com/uzh-rpg)
- Thanks to [Baptiste](https://github.com/Baptiste-AIST) for providing the wooden block reconstruction data from [[1](https://robocip-aist.github.io/sii_nerf_scans/)]

## References

[1] Erich, F., Bourreau, B., *Neural Scanning: Rendering and determining geometry of household objects using Neural Radiance Fields*. [Link](https://robocip-aist.github.io/sii_nerf_scans/). 2022

## Citation

If this work helps with your research, please cite:

```bibtex
@inproceedings{meyer2023cherrypicker,
  title={CherryPicker: Semantic skeletonization and topological reconstruction of cherry trees},
  author={Meyer, Lukas and Gilson, Andreas and Scholz, Oliver and Stamminger, Marc},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6244--6253},
  year={2023}
}
```
