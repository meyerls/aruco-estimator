#!/bin/bash

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda init bash

conda update -n base -c defaults conda -y
conda create -y -n aruco_estimator

conda activate aruco_estimator

conda install -y python==3.8
conda install -y pip

python3 -m pip install --upgrade pip
pip install -r requirements.txt

wget https://exiftool.org/Image-ExifTool-12.51.tar.gz
gzip -dc Image-ExifTool-12.51.tar.gz | tar -xf -
cd Image-ExifTool-12.51
perl Makefile.PL
make test
sudo make install
