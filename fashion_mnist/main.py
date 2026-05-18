import argparse
import json

import torch
import torch.nn as nn
from fashion_mnist import config, load_data
from fashion_mnist.model import SimpleCNN
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from utils import train_utils


def main(exp_name: str, use_early_stopping: bool = True, data_root: str | None = None):

    if data_root is None:
        dataset_config = config.DataConfig()
    else:
        dataset_config = config.DataConfig(root=data_root)

    exp_config = config.ExperimentConfig(name=exp_name, dataset=dataset_config)
    train_config = exp_config.training
    data_config = exp_config.dataset

    writer = SummaryWriter(f"{data_config.experiment_path}/{exp_name}")

    torch.manual_seed(exp_config.seed)

    train_data = load_data.load_dataset(
        data_config,
        train=True,
        transforms=v2.Compose(
            [
                v2.RandomHorizontalFlip(train_config.h_flip_prob),
                v2.RandomRotation(degrees=train_config.rotate_range),
                v2.RandomCrop(28, padding=2),
                load_data.ZeroOneScale(),
                v2.Normalize(mean=[train_config.mean], std=[train_config.std]),
            ]
        ),
    )
    test_data = load_data.load_dataset(
        data_config,
        train=False,
        transforms=v2.Compose(
            [
                load_data.ZeroOneScale(),
                v2.Normalize(mean=[train_config.mean], std=[train_config.std]),
            ]
        ),
    )

    train_dataloader = DataLoader(
        train_data, batch_size=train_config.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_data, batch_size=train_config.batch_size, shuffle=False
    )

    model = SimpleCNN(exp_config.model)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        compile_backend = "inductor"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        compile_backend = "aot_eager"
    else:
        device = torch.device("cpu")
        compile_backend = None
    print(f"Using device: {device}")

    train_utils.print_model_summary(
        model,
        device,
        (
            train_config.input_channels,
            train_config.input_height,
            train_config.input_width,
        ),
    )
    torch.compile(model, backend=compile_backend)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=train_config.scheduler_patience,
        factor=train_config.scheduler_factor,
    )
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    if use_early_stopping:
        early_stopping = train_utils.EarlyStoppingWithCheckpoint(
            model_path=data_config.model_path,
            model_name=exp_name,
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

    if not use_early_stopping:
        train_utils.save_model(model, data_config.model_path, f"{exp_name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name", type=str, default=train_utils.generate_default_exp_name()
    )
    parser.add_argument(
        "--early_stopping",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    main(args.exp_name, args.early_stopping, data_root=args.data_root)
