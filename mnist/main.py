import argparse

import config
import load_data
import torch
import torch.nn as nn
import train_utils
from model import SimpleCNN
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def main(exp_name: str):
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

    train_utils.train_many_epochs(
        train_config.num_epochs,
        train_dataloader,
        test_dataloader,
        model,
        loss_fn,
        optimizer,
        device,
        writer=writer,
    )

    writer.close()
    torch.save(model.state_dict(), f"{data_config.model_path}/{exp_name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)

    args = parser.parse_args()
    main(args.exp_name)
