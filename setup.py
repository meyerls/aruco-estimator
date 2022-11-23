#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
import setuptools
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='aruco-estimator',
    version='1.1.6',
    description='Aruco Scale Factor Estimation',
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Lukas Meyer',
    author_email='lukas.meyer@fau.de',
    url="https://github.com/meyerls/aruco-estimator",
    packages=['aruco_estimator'],
    install_requires=["numpy",
                      "colmap_wrapper",
                      "matplotlib",
                      "open3d",
                      "opencv-contrib-python",
                      "pyquaternion",
                      "pycolmap",
                      "tqdm",
                      "wget"],  # external packages as dependencies
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
