import argparse
import json
import os
from typing import cast

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from utils import train_utils

from fashion_mnist import config, constants, load_data
from fashion_mnist.model import SimpleCNN


def main(
    exp_name: str | None = None,
    config_path: str | None = None,
):
    exp_name = exp_name or train_utils.generate_default_exp_name()
    base_config = OmegaConf.structured(config.ExperimentConfig)

    if config_path is not None:
        yaml_config = OmegaConf.load(config_path)
        exp_config = cast(
            config.ExperimentConfig, OmegaConf.merge(base_config, yaml_config)
        )
    else:
        exp_config = cast(config.ExperimentConfig, base_config)

    train_config = exp_config.training
    data_config = exp_config.dataset

    train_utils.seed_everything(exp_config.seed)

    writer = SummaryWriter(f"{data_config.experiment_path}/{exp_name}")

    train_dataloader, test_dataloader = load_data.get_dataloaders(exp_config)

    model = SimpleCNN(
        constants.INPUT_CHANNELS, constants.INPUT_HEIGHT, exp_config.model
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.compile(model)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.compile(model, backend="aot_eager")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_utils.print_model_summary(
        model,
        (
            constants.INPUT_CHANNELS,
            constants.INPUT_HEIGHT,
            constants.INPUT_WIDTH,
        ),
    )

    optimizer, scheduler = train_utils.get_optimizer_and_scheduler(
        model=model,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        scheduler_patience=train_config.scheduler_patience,
        scheduler_factor=train_config.scheduler_factor,
    )
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

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
        "--exp-name",
        type=str,
        default=None,
        help="Optional experiment name. Defaults to timestamp.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "experiment.yaml"),
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()

    main(
        exp_name=args.exp_name,
        config_path=args.config,
    )
