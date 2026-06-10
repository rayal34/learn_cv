import argparse
import json
import os
from typing import cast

import torch
import torch.nn as nn
from models.cnn import SimpleCNN
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from utils import train_utils

from mnist import config, constants, load_data
from mnist.utils import get_optimizer_and_scheduler


def main(config_path: str):
    base_config = OmegaConf.structured(config.ExperimentConfig)

    yaml_config = OmegaConf.load(config_path)
    exp_config = cast(
        config.ExperimentConfig, OmegaConf.merge(base_config, yaml_config)
    )

    train_config = exp_config.training
    data_config = exp_config.dataset

    train_utils.seed_everything(exp_config.seed)

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

    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        exp_config=exp_config,
    )
    train_loss_fn = nn.CrossEntropyLoss(reduction="sum")
    eval_loss_fn = nn.CrossEntropyLoss(reduction="sum")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if exp_config.dry_run:
        train_config.num_epochs = 1
        writer = None
    else:
        writer = SummaryWriter(f"{data_config.experiment_path}/{exp_config.name}")
        writer.add_text("Architecture", str(model))
        if OmegaConf.is_config(exp_config):
            config_dict = OmegaConf.to_container(exp_config, resolve=True)
        else:
            config_dict = exp_config.to_dict()
        writer.add_text("Config", json.dumps(config_dict))

    model = train_utils.train_many_epochs(
        epochs=train_config.num_epochs,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        train_loss_fn=train_loss_fn,
        eval_loss_fn=eval_loss_fn,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
    )

    if writer is not None:
        writer.close()

    if exp_config.early_stopping is not None:
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
