import argparse
import json
import os
from typing import cast

import torch
import torch.nn as nn
from models.resnet import ResNet18
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from utils import train_utils
from utils.loss_functions import SoftCrossEntropyLoss

from cifar import config, constants, load_data
from cifar.utils import get_optimizer_and_scheduler


def run_profiler(
    train_dataloader,
    test_dataloader,
    model,
    train_loss_fn,
    eval_loss_fn,
    device,
    optimizer,
    scheduler,
    scheduler_update_freq,
    profiler_dir,
    num_epochs=5,
):

    train_utils.train_many_epochs(
        epochs=num_epochs,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        train_loss_fn=train_loss_fn,
        eval_loss_fn=eval_loss_fn,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_update_freq=scheduler_update_freq,
        early_stopping=None,
        writer=None,
        profiler_dir=profiler_dir,
    )


def run_training(
    exp_config,
    train_config,
    data_config,
    model,
    train_dataloader,
    test_dataloader,
    train_loss_fn,
    eval_loss_fn,
    device,
    optimizer,
    scheduler,
    scheduler_update_freq,
    early_stopping,
    writer,
):
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
        scheduler_update_freq=scheduler_update_freq,
        early_stopping=early_stopping,
        writer=writer,
    )

    if writer is not None:
        writer.close()

    if not train_config.early_stopping:
        train_utils.save_model(model, data_config.model_path, f"{exp_config.name}.pt")


def main(config_path: str, profile: bool = False):
    base_config = OmegaConf.structured(config.ExperimentConfig)
    yaml_config = OmegaConf.load(config_path)

    exp_config = cast(
        config.ExperimentConfig, OmegaConf.merge(base_config, yaml_config)
    )
    train_utils.seed_everything(exp_config.seed)

    train_config = exp_config.training
    data_config = exp_config.dataset
    model_config = exp_config.model

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

    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        exp_config=exp_config,
        train_dataloader=train_dataloader,
    )

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

    train_loss_fn = SoftCrossEntropyLoss(reduction="sum")
    eval_loss_fn = nn.CrossEntropyLoss(reduction="sum")

    if profile:
        profiler_dir = f"{data_config.experiment_path}/{exp_config.name}/profile"
        run_profiler(
            train_dataloader,
            test_dataloader,
            model,
            train_loss_fn,
            eval_loss_fn,
            device,
            optimizer,
            scheduler,
            train_config.scheduler_update_freq,
            profiler_dir,
            num_epochs=5,
        )
    else:
        run_training(
            exp_config,
            train_config,
            data_config,
            model,
            train_dataloader,
            test_dataloader,
            train_loss_fn,
            eval_loss_fn,
            device,
            optimizer,
            scheduler,
            train_config.scheduler_update_freq,
            early_stopping,
            writer,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "experiment.yaml"),
    )

    parser.add_argument(
        "--profile",
        type=bool,
        default=False,
    )

    args = parser.parse_args()

    main(config_path=args.config, profile=args.profile)
