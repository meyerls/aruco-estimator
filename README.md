<p align="center" width="100%">
    <img width="100%" src="https://github.com/meyerls/aruco-estimator/blob/dev/img/wood.png?raw=true">
</p>

# Automatic Estimation of the Scale Factor Based on Aruco Markers (Work in Progress!)

<a href="https://pypi.org/project/aruco-estimator/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/aruco-estimator"></a>
<a href="https://pypi.org/project/aruco-estimator/"><img alt="PyPI" src="https://img.shields.io/pypi/v/aruco-estimator"></a>
<a href="https://github.com/meyerls/aruco-estimator/actions"><img alt="GitHub Workflow Status" src="https://img.shields.io/github/workflow/status/meyerls/aruco-estimator/Python%20package"></a>
<a href="https://github.com/meyerls/aruco-estimator/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/github/license/meyerls/aruco-estimator"></a>
<!--a href="https://pepy.tech/project/aruco-estimator"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/aruco-estimator?label=PyPi%20downloads"></a--> 


<!--![PyPI](https://img.shields.io/pypi/v/aruco-estimator)
![PyPI - Downloads](https://img.shields.io/pypi/dm/aruco-estimator?label=PyPi%20downloads)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/meyerls/aruco-estimator/Publish%20Python%20%F0%9F%90%8D%20distributions%20%F0%9F%93%A6%20to%20PyPI%20and%20TestPyPI)
![GitHub](https://img.shields.io/github/license/meyerls/aruco-estimator)-->

## About

This project aims to automatically compute the correct scale of a point cloud generated
with [COLMAP](https://colmap.github.io/) by placing an aruco marker into the scene.

## Installation

### PyPi

This repository is tested on Python 3.6+ and can be installed from [PyPi](https://pypi.org/project/aruco-estimator)
<!-- Tutorial: https://www.youtube.com/watch?v=JkeNVaiUq_c -->

````angular2html
pip install aruco-estimator
````

## Usage

### Dataset

An exemplary data set is provided. The dataset shows a simple scene of a door with an aruco marker. Other dataset might
follow in future work. It can be downloaded by using

````python
from aruco_estimator import download

dataset = download.Dataset()
dataset.download_door_dataset(output_path='.')
````

### API

A use of the code on the provided dataset can be seen in the following block. The most important function is 
``aruco_scale_factor.run()``. Here, an aruco marker is searched for in each image. If a marker is found in at 
least 2 images, the position of the aruco corner in 3D is calculated based on the camera poses and the corners of 
the aruco maker.Based on the positions of the corners of the square aruco marker, the size of the marker in the unscaled 
reconstruction can be determined. With the correct metric size of the marker, the scene can be scaled true to scale 
using ``aruco_scale_factor.apply(true_scale)``. 

````python
from aruco_estimator.aruco_scale_factor import ArucoScaleFactor
from aruco_estimator.visualization import ArucoVisualization
from aruco_estimator import download
from colmap_wrapper.colmap import COLMAPProject
import os
import open3d as o3d

# Download example dataset. Door dataset is roughly 200 MB
dataset = download.Dataset()
dataset.download_door_dataset()

# Load Colmap project folder
project = COLMAPProject(project_path=dataset.dataset_path, image_resize=0.4)

# Init & run pose estimation of corners in 3D & estimate mean L2 distance between the four aruco corners
aruco_scale_factor = ArucoScaleFactor(photogrammetry_software=project, aruco_size=dataset.scale)
aruco_distance = aruco_scale_factor.run()
print('Size of the unscaled aruco markers: ', aruco_distance)

# Calculate scaling factor, apply it to the scene and save scaled point cloud
dense, scale_factor = aruco_scale_factor.apply() 
print('Point cloud and poses are scaled by: ', scale_factor)
print('Size of the scaled (true to scale) aruco markers in meters: ', aruco_distance * scale_factor)

# Visualization of the scene and rays 
vis = ArucoVisualization(aruco_colmap=aruco_scale_factor)
vis.visualization(frustum_scale=0.7, point_size=0.1)

# Write Data
aruco_scale_factor.write_data()
````

## Source

If you want to install the repo from source make sure that conda is installed. Afterwards clone this repository, give
the bash file executable rights and install the conda env:

````angular2html
git clone https://github.com/meyerls/aruco-estimator.git
cd aruco-estimator
chmod u+x init_env.sh
# You might need a password to install exiftools
./init_env.sh
````

To test the code on your local machine try the example project by using:

````angular2html
python3 scale_estimator.py --test_data
````
<p align="center" width="100%">
    <img width="100%" src="https://github.com/meyerls/aruco-estimator/blob/dev/img/door.png?raw=true">
</p>

<p align="center" width="100%">
    <img width="100%" src="https://github.com/meyerls/aruco-estimator/blob/dev/img/output.gif?raw=true">
</p>

## Limitation / Improvements

- [ ] Up to now only SIMPLE_RADIAL and PINHOLE camera models are supported. Extend all models
- [ ] Up to now only one aruco marker per scene can be detected. Multiple aruco marker could improve the scale
  estimation
- [ ] Different aruco marker settings and marker types should be investigated for different scenarios to make it either more robust to
  false detections
- [ ] Geo referencing of aruco markers with earth coordinate system using GPS or RTK
- [ ] Only COLMAP is supported. Add additional reconstruction software.

## Acknowledgement

* The Code to read out the binary COLMAP data is partly borrowed from the
repo [COLMAP Utility Scripts](https://github.com/uzh-rpg/colmap_utils) by [uzh-rpg](https://github.com/uzh-rpg).
* The visualization of the wooden block is created from the dataset found in [[1](https://robocip-aist.github.io/sii_nerf_scans/)]

## Trouble Shooting

*In some cases cv2 does not detect the aruco marker module. Reinstalling opencv-python and opencv-python-python might
  help [Source](https://stackoverflow.com/questions/45972357/python-opencv-aruco-no-module-named-cv2-aruco)
* [PyExifTool](https://github.com/sylikc/pyexiftool): A library to communicate with the [ExifTool](https://exiftool.org)
  command- application. If you have trouble installing it please refer to the PyExifTool-Homepage. 
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
Thanks to [Baptiste](https://github.com/Baptiste-AIST) for providing the data for the cube reconstruction. Source from:
[1] <div class="csl-entry">Erich, F., Bourreau, B., <i>Neural Scanning: Rendering and determining geometry of household objects using Neural Radiance Fields</i> <a href="https://robocip-aist.github.io/sii_nerf_scans/">Link</a>. 2022</div>

## Citation

Please cite this paper, if this work helps you with your research:

```
@InProceedings{ ,
  author="H",
  title="",
  booktitle="",
  year="",
  pages="",
  isbn=""
}
```