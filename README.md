<p align="center" width="100%">
    <img width="100%" src="https://media.githubusercontent.com/media/meyerls/aruco-estimator/main/img/wood.png">
</p>

# Automatic Estimation of the Scale Factor Based on Aruco Markers

<a href="https://pypi.org/project/aruco-estimator/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/aruco-estimator"></a>
<a href="https://pypi.org/project/aruco-estimator/"><img alt="PyPI" src="https://img.shields.io/pypi/v/aruco-estimator"></a>
<a href="https://github.com/meyerls/aruco-estimator/actions"><img alt="GitHub Workflow Status" src="https://img.shields.io/github/workflow/status/meyerls/aruco-estimator/Python%20package"></a>
<a href="https://github.com/meyerls/aruco-estimator/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/github/license/meyerls/aruco-estimator"></a>

## About

This project aims to automatically compute the correct scale of a point cloud generated with [COLMAP](https://colmap.github.io/) by placing an aruco marker into the scene.

## Installation

### PyPi

This repository is tested on Python 3.6+ and can be installed from [PyPi](https://pypi.org/project/aruco-estimator)

```bash
pip install aruco-estimator
```

However, the above is out of date and broken as of 2025-03-22.  Instead, you should install from source:


### From Source (Conda)

First, check that you have an NVIDIA GPU, because a GPU is required with CUDA for `colmap patch_match_stereo --workspace_path data\door\dense`.

```bash
# Check the max version your computer is CAPABLE of running: 
nvidia-smi
# Check the CUDA version you have installed:
nvcc --version
```

This version must be something higher than 11 I think, for the COLMAP with CUDA bindings below to work.  If you can upgrade, then do so first.  If you cannot, then you won't be able to do dense reconstructions and therefore you won't be able to use this library.

Next, install COLMAP with CUDA bindings.  Here are the commands for Windows Powershell:

```powershell
rd /s /q C:\colmap

curl -L -o C:\colmap.zip https://github.com/colmap/colmap/releases/download/3.11.1/colmap-x64-windows-cuda.zip

:: Extract the ZIP file to C:\colmapcommand
powershell -Command "Expand-Archive -Path C:\colmap.zip -DestinationPath C:\colmap"

:: Update the PATH for the current session
set PATH=C:\colmap\bin;%PATH%
```

Finally, install the aruco-estimator repo and run the example script:

```bash
git clone https://github.com/MichaelCurrie/aruco-estimator.git
cd aruco-estimator
conda create -n arenv python=3.9 -y
conda activate arenv
conda install -c conda-forge exiftool -y
pip install .
python example.py
```

## Usage

### Scale Factor Estimation Example

A runnable example with a dataset taken from a door is available by running:

```bash
python example.py
```

### Registration and Scaling

In some cases COLMAP is not able to "registrate" all images into one dense reconstruction. If appears to be reconstructed into two seperated reconstruction. To registrate both (for now, only two are possible) reconstructions the ArUco markers are used to registrate both sides using ```ArucoRegistration```.

```python
from aruco_estimator.registration import ArucoRegistration

scaled_registration = ArucoRegistration(
    project_path_a=[path2part1],
    project_path_b=[path2part2]
)
scaled_registration.scale(debug=True)
scaled_registration.registrate(manual=False, debug=True)
scaled_registration.write()
```

To test the code on your local machine try the example project by using:

````angular2html
python3 aruco_estimator/test.py --test_data --visualize --frustum_size 0.4
````
<p align="center" width="100%">
    <img width="100%" src="https://github.com/meyerls/aruco-estimator/blob/main/img/door.png?raw=true">
</p>

<p align="center" width="100%">
    <img width="100%" src="https://github.com/meyerls/aruco-estimator/blob/main/img/output.gif?raw=true">
</p>

## Limitations / Improvements

- [ ] Up to now only SIMPLE_RADIAL and PINHOLE camera models are supported. Extend all models
- [ ] Up to now only one aruco marker per scene can be detected. Multiple aruco marker could improve the scale estimation
- [ ] Different aruco marker settings and marker types should be investigated for different scenarios to make it either more robust to false detections
- [ ] Geo referencing of aruco markers with earth coordinate system using GPS or RTK
- [ ] Only COLMAP is supported. Add additional reconstruction software.

## Acknowledgements

* The code to read out the binary COLMAP data is partly borrowed from the repo [COLMAP Utility Scripts](https://github.com/uzh-rpg/colmap_utils) by [uzh-rpg](https://github.com/uzh-rpg).
* Thanks to [Baptiste](https://github.com/Baptiste-AIST) for providing the data for the wooden block reconstruction. Source from [[1](https://robocip-aist.github.io/sii_nerf_scans/)]

## Troubleshooting

* In some cases cv2 does not detect the aruco marker module. Reinstalling opencv-python and opencv-python-python might help [Source](https://stackoverflow.com/questions/45972357/python-opencv-aruco-no-module-named-cv2-aruco)
* [PyExifTool](https://github.com/sylikc/pyexiftool): A library to communicate with the [ExifTool](https://exiftool.org) application. If you have trouble installing it please refer to the PyExifTool-Homepage. 
```bash
# For Ubuntu users:
wget https://exiftool.org/Image-ExifTool-12.51.tar.gz
gzip -dc Image-ExifTool-12.51.tar.gz | tar -xf -
cd Image-ExifTool-12.51
perl Makefile.PL
make test
sudo make install
```

## References
<div class="csl-entry">[1] Erich, F., Bourreau, B., <i>Neural Scanning: Rendering and determining geometry of household objects using Neural Radiance Fields</i> <a href="https://robocip-aist.github.io/sii_nerf_scans/">Link</a>. 2022</div>

## Citation

Please cite this paper, if this work helps you with your research:

```
@InProceedings{ ,
  author="",
  title="",
  booktitle="",
  year="",
  pages="",
  isbn=""
}
```