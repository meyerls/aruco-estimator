#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='aruco_estimator',
    version='1.0.5',
    description='Aruco Scale Factor Estimation',
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Lukas Meyer',
    author_email='lukas.meyer@fau.de',
    url="https://github.com/meyerls/aruco-estimator",
    packages=setuptools.find_packages(),
    install_requires=["numpy==1.22.4",
                      "matplotlib==3.5.2",
                      "open3d==0.15.2",
                      "opencv-python==4.6.0.66",
                      "opencv-contrib-python==4.6.0.66",
                      "pyquaternion==0.9.9",
                      "tqdm==4.64.0",
                      "wget==3.2"],  # external packages as dependencies
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
