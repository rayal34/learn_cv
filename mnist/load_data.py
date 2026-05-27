from base import config
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor

from mnist import constants


def load_training_data(root):

    return datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=Compose(
            [ToTensor(), Normalize(mean=constants.MEAN, std=constants.STD)]
        ),
    )


def load_test_data(root):
    return datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=Compose(
            [ToTensor(), Normalize(mean=constants.MEAN, std=constants.STD)]
        ),
    )


def get_dataloaders(config: config.ExperimentConfig):
    train_data = load_training_data(config.dataset.root)
    test_data = load_test_data(config.dataset.root)

    train_dataloader = DataLoader(
        train_data,
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_data, batch_size=config.training.batch_size, shuffle=False
    )

    return train_dataloader, test_dataloader
