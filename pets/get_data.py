import argparse
import logging
import os
import tarfile

import requests
from omegaconf import OmegaConf

from pets.config import DataConfig

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

IMAGES_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
ANNOTATIONS_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"


def download_tar(url, destination_dir):
    logging.info(f"Downloading {url}")
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)

    filename = url.split("/")[-1]
    destination_path = os.path.join(destination_dir, filename)
    resp = requests.get(url)
    resp.raise_for_status()
    with open(destination_path, "wb") as f:
        f.write(resp.content)

    return destination_path


def untar(tar_path, destination):
    logging.info(f"Extracting {tar_path}")
    if not os.path.exists(tar_path):
        return
    tar = tarfile.open(tar_path, "r:gz")
    tar.extractall(path=destination)
    tar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "experiment.yaml"),
    )
    args = parser.parse_args()

    cfg = OmegaConf.structured(DataConfig)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config).dataset)

    save_dir = cfg.root
    for url in [IMAGES_URL, ANNOTATIONS_URL]:
        tar_path = download_tar(url, save_dir)
        untar(tar_path, save_dir)
