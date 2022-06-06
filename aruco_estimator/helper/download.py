#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
# ...

# Libs
import os
import wget
from zipfile import ZipFile
from tqdm import tqdm
import urllib.request


# Own modules
# ...

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract_archive(url: str, output_directory: str, overwrite: bool = False) -> None:
    if output_directory == os.path.abspath(__file__):
        data_path = os.path.join(output_directory, '..', '..', '..', 'data')
    else:
        data_path = os.path.join(output_directory, 'data')

    os.makedirs(data_path, exist_ok=True)

    filename = os.path.join(data_path, url.split('/')[-1])

    if os.path.exists(filename) and not overwrite:
        print('{} already exists in {}'.format(url.split('/')[-1], data_path))
    else:
        with DownloadProgressBar(unit='B',
                                 unit_scale=True,
                                 miniters=1,
                                 desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)

    # opening the zip_file file in READ mode
    with ZipFile(filename, 'r') as zip_file:
        # printing all the contents of the zip_file file
        # zip_file.printdir()

        # extracting all the files
        print('Extracting all the files now...')
        zip_file.extractall(path=data_path)
        print('Done!')

    return os.path.join(data_path, url.split('/')[-1].split('.zip')[0])


Datasets = {'door': 'https://faubox.rrze.uni-erlangen.de/dl/fiUNWMmsaEAavXHfjqxfyXU9/door.zip'}


def download_door_dataset(output_path: str = os.path.abspath(__file__)):
    return download_and_extract_archive(url=Datasets['door'], output_directory=output_path)


if __name__ == '__main__':
    download_door_dataset()
