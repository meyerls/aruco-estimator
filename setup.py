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
    version='1.0.6',
    description='Aruco Scale Factor Estimation',
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Lukas Meyer',
    author_email='lukas.meyer@fau.de',
    url="https://github.com/meyerls/aruco-estimator",
    packages=setuptools.find_packages(),
    install_requires=["numpy",
                      "matplotlib",
                      "open3d",
                      "opencv-python",
                      "opencv-contrib-python",
                      "pyquaternion",
                      "tqdm",
                      "wget"],  # external packages as dependencies
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
