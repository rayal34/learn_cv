import argparse
import json
import os
from typing import cast

import torch
import torch.nn as nn
from base.model import SimpleCNN
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from utils import train_utils

from mnist import config, constants, load_data


def main(config_path: str):
    base_config = OmegaConf.structured(config.ExperimentConfig)

    yaml_config = OmegaConf.load(config_path)
    exp_config = cast(
        config.ExperimentConfig, OmegaConf.merge(base_config, yaml_config)
    )

    train_config = exp_config.training
    data_config = exp_config.dataset

    train_utils.seed_everything(exp_config.seed)
    writer = SummaryWriter(f"{data_config.experiment_path}/{exp_config.name}")

    train_dataloader, test_dataloader = load_data.get_dataloaders(exp_config)

    model = SimpleCNN(
        constants.INPUT_CHANNELS, constants.INPUT_HEIGHT, exp_config.model
    )
    torch.compile(model, backend="aot_eager")
    train_utils.print_model_summary(
        model,
        (
            constants.INPUT_CHANNELS,
            constants.INPUT_HEIGHT,
            constants.INPUT_WIDTH,
        ),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=train_config.scheduler_patience,
        factor=train_config.scheduler_factor,
    )
    loss_fn = nn.CrossEntropyLoss()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if train_config.early_stopping:
        early_stopping = train_utils.EarlyStoppingWithCheckpoint(
            model_path=data_config.model_path,
            model_name=exp_config.name,
            patience=train_config.early_stopping_patience,
        )
    else:
        early_stopping = None

    train_utils.train_many_epochs(
        train_config.num_epochs,
        train_dataloader,
        test_dataloader,
        model,
        loss_fn,
        device,
        optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        writer=writer,
    )

    writer.add_text("Architecture", str(model))
    writer.add_text("Config", json.dumps(exp_config.to_dict()))
    writer.close()

    if not train_config.early_stopping:
        train_utils.save_model(model, data_config.model_path, f"{exp_config.name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "experiment.yaml"),
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()

    main(config_path=args.config)
