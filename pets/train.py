import argparse
import json
import os
from typing import cast

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet50_Weights, resnet50

from core.train_utils import (
    EarlyStoppingWithCheckpoint,
    print_model_summary,
    save_model,
    seed_everything,
)
from models import constants as model_constants
from models.object_detection import ObjectDetectionFromResnet
from pets import config, constants
from pets.utils import data, fine_tuning, training

OmegaConf.register_new_resolver(
    "constants", lambda name: getattr(constants, name), replace=True
)


OmegaConf.register_new_resolver(
    "model_constants", lambda name: getattr(model_constants, name), replace=True
)


def main(config_path: str, profile: bool = False):

    base_config = OmegaConf.structured(config.ExperimentConfig)
    yaml_config = OmegaConf.load(config_path)
    exp_config = cast(
        config.ExperimentConfig, OmegaConf.merge(base_config, yaml_config)
    )

    seed_everything(exp_config.seed)

    train_config = exp_config.training
    data_config = exp_config.dataset

    os.environ["TORCH_HOME"] = data_config.torch_home
    train_dataloader, test_dataloader = data.get_dataloaders(exp_config)

    model = ObjectDetectionFromResnet(
        backbone=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        num_classes=constants.NUM_CLASSES,
        model_config=exp_config.model,
    )

    if exp_config.fine_tune_freezing_strategy is not None:
        freezing_func = getattr(
            fine_tuning, exp_config.fine_tune_freezing_strategy.type
        )

        model = freezing_func(model, **exp_config.fine_tune_freezing_strategy.params)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = cast(nn.Module, torch.compile(model))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = cast(nn.Module, torch.compile(model, backend="aot_eager"))
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print_model_summary(
        model,
        (
            constants.INPUT_CHANNELS,
            model_constants.RESNET_INPUT_SIZE,
            model_constants.RESNET_INPUT_SIZE,
        ),
    )

    optimizer, scheduler = training.get_optimizer_and_scheduler(
        model=model,
        exp_config=exp_config,
        train_dataloader=train_dataloader,
    )

    if exp_config.early_stopping:
        early_stopping = EarlyStoppingWithCheckpoint(
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

    num_epochs = train_config.num_epochs
    if profile:
        profiler_dir = f"{data_config.experiment_path}/{exp_config.name}/profile"
        training.profile(
            dataloader=train_dataloader,
            model=model,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            profiler_dir=profiler_dir,
        )
        return

    model = training.train_many_epochs(
        epochs=num_epochs,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_update_freq=exp_config.scheduler.update_freq,
        early_stopping=early_stopping if not profile else None,
        writer=writer,
    )

    if writer is not None:
        writer.close()

    if not profile and early_stopping is None:
        save_model(model, data_config.model_path, f"{exp_config.name}.pt")


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
