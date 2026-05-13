from learn_cv.mnist import config
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor


def load_training_data(root):

    mean, std = get_normalize_mean_std()
    return datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=Compose([ToTensor(), Normalize(mean=mean, std=std)]),
    )


def load_test_data(root):
    mean, std = get_normalize_mean_std()
    return datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=Compose([ToTensor(), Normalize(mean=mean, std=std)]),
    )


def get_normalize_mean_std():
    data_config = config.DataConfig()
    train_dataloader = DataLoader(
        datasets.MNIST(
            root=data_config.data_path,
            train=True,
            download=True,
            transform=ToTensor(),
        ),
        batch_size=10000,
        drop_last=False,
    )

    mean, std = 0.0, 0.0
    for X, _ in train_dataloader:
        mean += X.mean()
        std += X.std()

    mean /= len(train_dataloader)
    std /= len(train_dataloader)

    return mean, std
