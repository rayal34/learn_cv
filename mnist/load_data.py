import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.io import decode_image
from torchvision.transforms import Lambda, ToTensor


def load_training_data(root):
    return datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=ToTensor(),
    )


def load_test_data(root):
    return datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=ToTensor(),
    )


class MNISTDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
