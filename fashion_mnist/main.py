import argparse
import json
import os
from typing import cast

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from core import io, train_utils
from fashion_mnist import config, constants
from fashion_mnist.utils import load_data, training
from models.cnn import SimpleCNN


def main(config_path: str):
    base_config = OmegaConf.structured(config.ExperimentConfig)

    yaml_config = OmegaConf.load(config_path)
    exp_config = cast(
        config.ExperimentConfig, OmegaConf.merge(base_config, yaml_config)
    )

    train_config = exp_config.training
    data_config = exp_config.dataset
    model_config = exp_config.model

    train_utils.seed_everything(exp_config.seed)

    train_dataloader, test_dataloader = load_data.get_dataloaders(exp_config)

    model = SimpleCNN(constants.INPUT_CHANNELS, constants.INPUT_HEIGHT, model_config)
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

    optimizer, scheduler = training.get_optimizer_and_scheduler(
        model=model,
        exp_config=exp_config,
    )
    train_loss_fn = nn.CrossEntropyLoss(reduction="sum")
    eval_loss_fn = nn.CrossEntropyLoss(reduction="sum")

    if exp_config.early_stopping:
        early_stopping = train_utils.EarlyStoppingWithCheckpoint(
            model_path=data_config.model_path,
            model_name=exp_config.name,
            patience=exp_config.early_stopping.patience,
            min_delta=exp_config.early_stopping.min_delta,
            higher_is_better=exp_config.early_stopping.higher_is_better,
        )
    else:
        early_stopping = None

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

    model = training.train_many_epochs(
        epochs=train_config.num_epochs,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        train_loss_fn=train_loss_fn,
        eval_loss_fn=eval_loss_fn,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        writer=writer,
    )

    if writer is not None:
        writer.close()

    if exp_config.early_stopping is not None:
        io.save_model(model, data_config.model_path, f"{exp_config.name}.pt")


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
