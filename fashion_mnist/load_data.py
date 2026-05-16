import gzip

import numpy as np
import torch
from fashion_mnist import config
from torch.utils.data import Dataset


def load_images(filename: str):
    with gzip.open(filename, "rb") as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    images = images.reshape(-1, 28, 28)
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

        image = torch.tensor(np.expand_dims(image, 0), dtype=torch.float32)

        if self.transforms:
            image = self.transforms(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label


class ZeroOneScale:
    def __init__(self, min=0.0, max=255.0):
        self.min = min
        self.max = max

    def __call__(self, img):
        return (img - self.min) / (self.max - self.min)
