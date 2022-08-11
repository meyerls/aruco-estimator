![](https://media.githubusercontent.com/media/meyerls/aruco-estimator/main/img/header.png)

# Aruco Scale Factor Estimation for COLMAP (Work in Progress!)

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

This project aims to automatically compute the correct scale of a point cloud scene generated
with  [COLMAP](https://colmap.github.io/) (we have only a colmap parser). by adding an aruco marker with into the scene.
Due to the known scale is possible to overcome the scale ambiguity from monocular SfM and SLAM. In the literature it is
commonly suggested to manually measure a known distance of a calibration object in the 3D reconstruction and then scale
the reconstruction accordingly or make photos from two known positions[1]. Since there is unfortunately no automated
approach to calculate an accurate scaling factor, this project aims to be attached as a post processing step to the
reconstruction process.

## Setup

Before taking the images for the reconstruction, an aruco marker must be printed out. Care should be taken to ensure
that the size of the marker fits the scene to be recognizable in the images. Additionally the marker is placed on a
planar surface and does not have curvature (otherwise, the marker will be recognized incorrectly or not at all). The
aruco marker can either be generated
with [ ArUco-markers-with-OpenCV-and-Python](https://github.com/KhairulIzwan/ArUco-markers-with-OpenCV-and-Python) or
generated easily on [this website](https://chev.me/arucogen/). When taking the images, make sure that the aruco marker
is visible in at least two images; otherwise, no solution exists. It is advisable to have at least five images with the
aruco marker for an accurate solution (two pictures should already be enough for a sufficient approximation. However,
blurry images or aruco markers that are too small could falsify the result). After the images have been acquired, the
reconstructions process can be carried out using COLMAP.

<!-- ## Theory

At first the extrinsic <img src="https://render.githubusercontent.com/render/math?math=\mathbf{M}_i"> and intrinsic
paramters <img src="https://render.githubusercontent.com/render/math?math=\mathbf{K}_i"> for every
image <img src="https://render.githubusercontent.com/render/math?math=\mathbf{I}_i"> of the reconstruction are readout
from the parsed COLMAP project and their underlying [binary output format](https://colmap.github.io/format.html). Then,
in each image <img src="https://render.githubusercontent.com/render/math?math=\mathbf{I}_i">, it is checked whether an
Arco marker is present. If so, all four corners of the square Aruco markers are extracted as image
coordinates <img src="https://render.githubusercontent.com/render/math?math=\mathbf{x} = (x_i, y_i, 1)^\top">. Thus it
is possible to cast a ray from the origin of the camera center
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{C}_i">
through all four Aruco corners trough two points according to
<img src="https://render.githubusercontent.com/render/math?math=r_{vec} = \mathbf{C}_i + ||\mathbf{x} \mathbf{K}^{-1} ||_2^2 \mathbf{R}_i \lambda">
with <img src="https://render.githubusercontent.com/render/math?math=\lambda \in [-\infty, +\infty]">  . Here x is the
image coordinate in homogeneous coordinates, K^-1 is the inverse matrix of the intrinsic matrix of the camera, R_i the
rotation of the extrinsic camera parameters and C_i the translation t_i of the camera pose. Four rays are cast for each
image in which an aruco marker is detected. Thus, with a minimum of two rays per aruco corner, the position in 3D space
can be determined trough the intersection of lines. The intersection of several 3D lines can then be calculated using a
least-squares method [2].-->

## Installation

### PyPi

This repository is tested on Python 3.6+ and can be installed from [PyPi](https://pypi.org/project/aruco-estimator)
<!-- Tutorial: https://www.youtube.com/watch?v=JkeNVaiUq_c -->

````angular2html
pip install aruco-estimator
````

### Source

````angular2html
git clone --recurse-submodules -j8 https://github.com/meyerls/aruco-estimator.git
````

## Usage

### Command Line

````angular2html
usage: scale_estimator.py [-h] [--colmap_project COLMAP_PROJECT]  [--visualize VISUALIZE]
Estimate scale factor for COLMAP projects with aruco markers.

optional arguments:
-h, --help                        show this help message and exit
--colmap_project COLMAP_PROJECT   Path to COLMAP project
--visualize VISUALIZE             Flag to enable visualization
````

To test the code on your local machine try the example project by using:

````angular2html
python scale_estimator.py- -test_data
````

### API

````python
from aruco_estimator.aruco_scale_factor import ArucoScaleFactor
from aruco_estimator import download

# Download example dataset. Door dataset is roughly 200 MB
dataset = download.Dataset()
dataset.download_door_dataset()

# Init & run pose estimation of corners in 3D & estimate mean L2 distance between the four aruco corners
aruco_scale_factor = ArucoScaleFactor(project_path=dataset.dataset_path)
aruco_distance = aruco_scale_factor.run()
print('Mean distance between aruco markers: ', aruco_distance)

# Calculate scaling factor, apply it to the scene and save scaled point cloud
dense, scale_factor = aruco_scale_factor.apply(true_scale=dataset.scale)  # scale in cm
print('Point cloud and poses are scaled by: ', scale_factor)

# Visualization of the scene and rays BEFORE scaling. This might be necessary for debugging
aruco_scale_factor.visualize_estimation(frustum_scale=0.2)
````

### Visualization

The visualization is suitable for displaying the scene, the camera frustums, and the casted rays. It is usefull to check
whether the corners of the aruco marker are detected and positioned correctly. If the aruco markers are not detected
correctly the aruco settings must be changed according to the scene to be more robust.

![](https://media.githubusercontent.com/media/meyerls/aruco-estimator/main/img/visualization.png)

## Limitation/Improvements

- [x] Make package for PyPi
- [x] Upload to PyPi during CI
- [x] Make dataset available for download
- [x] Put aruco marker detection in threads
- [ ] Install CLI Tool vi PyPi
- [ ] Scale poses of camera/extrinsics.
- [ ] Up to now only one aruco marker per scene can be detected. Multiple aruco marker could improve the scale
  estimation
- [ ] Different aruco marker settings should be investigated for different scenarios to make it either more robust to
  false detections
- [ ] Geo referencing of aruco markers with earth coordinate system using GPS or RTK
- [ ] Alternatives to aruco marker should be investigated.
- [ ] Are the corners from the aruco marker returned identical regarding the orientation in the image?

## Acknowledgement

The Code to read out the binary COLMAP data is partly borrowed from the
repo [COLMAP Utility Scripts](https://github.com/uzh-rpg/colmap_utils) by [uzh-rpg](https://github.com/uzh-rpg).

## Trouble Shooting

- In some cases cv2 does not detect the aruco marker module. Reinstalling opencv-python and opencv-python-python might
  help [Source](https://stackoverflow.com/questions/45972357/python-opencv-aruco-no-module-named-cv2-aruco)

## Resources

<div class="csl-entry">[1] Lourakis, M., &#38; Zabulis, X. (n.d.). <i>LNCS 8047 - Accurate Scale Factor Estimation in 3D Reconstruction</i>. 2013</div>
<div class="csl-entry">[2] Traa, J., <i>Least-Squares Intersection of Lines</i>. 2013</div>
