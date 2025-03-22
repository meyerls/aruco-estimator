"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

import logging
import os
import urllib.request
from zipfile import ZipFile
import hashlib
from pathlib import Path

import wget
from tqdm import tqdm

DOOR_DATASET = {
    "url": "https://faubox.rrze.uni-erlangen.de/dl/fiUNWMmsaEAavXHfjqxfyXU9/door.zip",
    "file_hash": "7ccb0d0255519854158a555ac13a1efab95d77b45858a86f6725a4e2ab94a5a1",
    "data_path": Path("data") / "door",
    "zip_path": Path("data") / "door.zip",
    "scale": 0.15,
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url: str, output_dir: str, overwrite: bool = False):
    filename = os.path.join(output_dir, url.split("/")[-1])

    if os.path.exists(filename) and not overwrite:
        logging.info("{} already exists in {}".format(url.split("/")[-1], output_dir))
    else:
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)

    return filename


def extract(filename: str, output_dir: str):
    # opening the zip_file file in READ mode
    with ZipFile(filename, "r") as zip_file:
        # printing all the contents of the zip_file file
        # zip_file.printdir()

        # extracting all the files
        logging.info("Extracting all the files now...")
        zip_file.extractall(path=output_dir)
        logging.info("Done!")


class Dataset:
    def __init__(self):
        self.dataset_name = None
        self.dataset_path = None
        self.filename = None
        self.data_path = None
        self.scale = None  # in cm

    def download_door_dataset(self):
        self.download_dataset(**DOOR_DATASET)

    def download_dataset(
        self, data_path: str, zip_path: str, url: str, file_hash: str, scale: float
    ):
        zip_path = Path(zip_path)
        self.data_path = Path(data_path)
        self.dataset_path = str(Path(data_path).resolve())
        self.dataset_name = self.data_path.stem
        self.scale = scale  # m

        if (not zip_path.is_file()) or (not self.data_path.is_dir()):
            self.filename = download(url=url, output_dir=self.data_path, overwrite=True)
            cur_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest()
            assert file_hash == cur_hash, f"ZIP hash of {url} doesn't match {zip_path}"
            extract(filename=self.filename, output_dir=data_path)

        return self.dataset_path


if __name__ == "__main__":
    downloader = Dataset()
    downloader.download_door_dataset()

    logging.info("Saved at {}".format(downloader.dataset_path))
