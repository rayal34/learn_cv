import argparse
import logging
import os
from typing import cast

from omegaconf import OmegaConf

from pets.config import DataConfig
from pets.utils.data import create_test_datasets, create_trainval_datasets

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "experiment.yaml"),
    )

    args = parser.parse_args()
    cfg = OmegaConf.structured(DataConfig)
    cfg = cast(DataConfig, OmegaConf.merge(cfg, OmegaConf.load(args.config).dataset))

    create_trainval_datasets(cfg)
    create_test_datasets(cfg)
