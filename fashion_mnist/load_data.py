import gzip

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from fashion_mnist import config, constants


class FashionMNISTDataset(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        image = torch.tensor(image, dtype=torch.float32)

        if self.transforms:
            image = self.transforms(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label


class ZeroOneScale:
    def __init__(
        self, min_val: float = 0.0, max_val: float = constants.MAX_PIXEL_VALUE
    ):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        return (img - self.min_val) / (self.max_val - self.min_val)


def load_images(filename: str):
    with gzip.open(filename, "rb") as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    images = images.reshape(
        -1, constants.INPUT_CHANNELS, constants.INPUT_HEIGHT, constants.INPUT_WIDTH
    )
    return images


def load_labels(filename: str):
    with gzip.open(filename, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels


def load_dataset(config: config.DataConfig, train: bool = True, transforms=None):
    if train:
        images = load_images(f"{config.data_path}/{config.train_images_filename}")
        labels = load_labels(f"{config.data_path}/{config.train_labels_filename}")
    else:
        images = load_images(f"{config.data_path}/{config.test_images_filename}")
        labels = load_labels(f"{config.data_path}/{config.test_labels_filename}")

    return FashionMNISTDataset(images, labels, transforms)


def get_dataloaders(config: config.ExperimentConfig):
    train_data = load_dataset(
        config.dataset,
        train=True,
        transforms=v2.Compose(
            [
                v2.RandomHorizontalFlip(config.data_augmentations.h_flip_prob),
                v2.RandomRotation(degrees=config.data_augmentations.rotate_range),
                v2.RandomCrop(
                    constants.INPUT_HEIGHT,
                    padding=config.data_augmentations.crop_padding,
                ),
                ZeroOneScale(),
                v2.Normalize(mean=[constants.MEAN], std=[constants.STD]),
            ]
        ),
    )

    test_data = load_dataset(
        config.dataset,
        train=False,
        transforms=v2.Compose(
            [
                ZeroOneScale(),
                v2.Normalize(mean=[constants.MEAN], std=[constants.STD]),
            ]
        ),
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_data, batch_size=config.training.batch_size, shuffle=False
    )

    return train_dataloader, test_dataloader
