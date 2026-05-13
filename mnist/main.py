import argparse
import json
from datetime import datetime

import config
import load_data
import torch
import torch.nn as nn
import train_utils
from model import EarlyStoppingWithCheckpoint, SimpleCNN
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def main(exp_name: str, use_early_stopping: bool = True):
    exp_config = config.ExperimentConfig(name=exp_name)
    train_config = exp_config.training
    data_config = exp_config.dataset

    writer = SummaryWriter(f"{data_config.experiment_path}/{exp_name}")

    torch.manual_seed(exp_config.seed)

    train_data = load_data.load_training_data(data_config.data_path)
    test_data = load_data.load_test_data(data_config.data_path)

    train_dataloader = DataLoader(
        train_data, batch_size=train_config.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_data, batch_size=train_config.batch_size, shuffle=False
    )

    model = SimpleCNN(exp_config.model)
    optimizer = torch.optim.SGD(model.parameters(), lr=train_config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if use_early_stopping:
        early_stopping = EarlyStoppingWithCheckpoint(
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
        optimizer,
        device,
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
        "--exp-name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    parser.add_argument(
        "--early_stopping",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    main(args.exp_name, args.early_stopping)
