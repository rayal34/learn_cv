import argparse
import json
import os
from functools import partial
from typing import cast

import torch
import torch.nn as nn
from base import config
from models.resnet import ResNet18
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from utils import loss_functions, train_utils

from cifar import constants, load_data


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

    model = ResNet18(constants.INPUT_CHANNELS, constants.NUM_CLASSES, model_config)
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

    if train_config.early_stopping:
        early_stopping = train_utils.EarlyStoppingWithCheckpoint(
            model_path=data_config.model_path,
            model_name=exp_config.name,
            patience=train_config.early_stopping_patience,
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

    if exp_config.data_augmentations.mixup_alpha > 0.0:
        train_loop_fn = partial(
            train_utils.train_loop_with_mixup,
            alpha=exp_config.data_augmentations.mixup_alpha,
            num_classes=constants.NUM_CLASSES,
        )
        train_loss_fn = loss_functions.SoftCrossEntropyLoss(reduction="sum")
    else:
        train_loop_fn = train_utils.train_loop
        train_loss_fn = nn.CrossEntropyLoss(reduction="sum")

    eval_loss_fn = nn.CrossEntropyLoss(reduction="sum")

    if exp_config.training.use_profiler:
        profiler_dir = f"{data_config.experiment_path}/{exp_config.name}/profile"
    else:
        profiler_dir = None

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
        early_stopping=early_stopping,
        writer=writer,
        train_loop_fn=train_loop_fn,
        profiler_dir=profiler_dir,
    )

    if writer is not None:
        writer.close()

    if not train_config.early_stopping:
        train_utils.save_model(model, data_config.model_path, f"{exp_config.name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "experiment.yaml"),
    )

    args = parser.parse_args()

    main(config_path=args.config)
