<p align="center" width="100%">
    <img width="100%" src="assets/wood.png">
</p>

# Automatic Scale Factor Estimation Based on ArUco Markers

<a href="https://pypi.org/project/aruco-estimator/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/aruco-estimator"></a>
<a href="https://pypi.org/project/aruco-estimator/"><img alt="PyPI" src="https://img.shields.io/pypi/v/aruco-estimator"></a>
<a href="https://github.com/meyerls/aruco-estimator/actions"><img alt="GitHub Workflow Status" src="https://img.shields.io/github/workflow/status/meyerls/aruco-estimator/Python%20package"></a>
<a href="https://github.com/meyerls/aruco-estimator/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/github/license/meyerls/aruco-estimator"></a>

## About

This project automatically computes the correct scale and registration of 3D reconstructions generated with [COLMAP](https://colmap.github.io/) by placing one or more ArUco markers in the scene.

## Installation

This repository is tested on Python 3.6+ and can be installed from PyPI:

```bash
pip install aruco-estimator
```

**Note:** The PyPI release is currently out of date and broken as of 2025-03-22. For the latest stable version, install directly from source:

```bash
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
aruco-estimator register ./door --target-id 7 --show
```

<p align="center" width="100%">
    <img width="100%" src="assets/door.png?raw=true">
</p>

<p align="center" width="100%">
    <img width="100%" src="assets/output.gif?raw=true">
</p>

### Merging by Tag
 blah

### Scripting

``` python 
from aruco_estimator.sfm.colmap import COLMAPProject
project_path = Path(project)

# Load COLMAP project using new interface
logging.info("Loading COLMAP project...")
project = COLMAPProject(str(project_path))

# Store original project state if needed for visualization
original_project = None
if show_original:
    original_project = deepcopy(project)

# Run ArUco detection
logging.info("Detecting ArUco markers...")
aruco_localizer = ArucoLocalizer(
    project=project,
    dict_type=dict_type,
    target_id=target_id,
)
aruco_distance, aruco_corners_3d = aruco_localizer.run()
logging.info(f"Target ArUco ID: {target_id}")
logging.info(f"ArUco 3d points: {aruco_corners_3d}")
logging.info(f"ArUco marker distance: {aruco_distance}")
```

## Known Limitations 

- Dense cloud visualization and modification is currently broken
- Only SIMPLE_RADIAL and PINHOLE camera models are supported
- Aruco boards are not uniquely supported 
- Pose estimation is not robust to false detections; filtering would be beneficial
- Only COLMAP .bin and .txt models are supported

## Roadmap
- [x] Replace get_normalization_transform with kabsch_umeyama
- [ ] Project base should own localizer
- [ ] Implement the merge by tag tool 
- [ ] Support for additional camera models
- [ ] Improved pose estimation robustness
- [ ] Dense cloud visualization fixes
- [ ] Geo-referencing of ArUco markers with Earth coordinate system using GPS or RT

## Troubleshooting

### OpenCV ArUco Module Issues

If cv2 doesn't detect the ArUco marker module, try reinstalling OpenCV:

```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python opencv-contrib-python
```

### ExifTool Installation

This project uses [PyExifTool](https://github.com/sylikc/pyexiftool) to communicate with [ExifTool](https://exiftool.org). If you encounter installation issues:

**Ubuntu/Debian:**
```bash
wget https://exiftool.org/Image-ExifTool-12.51.tar.gz
gzip -dc Image-ExifTool-12.51.tar.gz | tar -xf -
cd Image-ExifTool-12.51
perl Makefile.PL
make test
sudo make install
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