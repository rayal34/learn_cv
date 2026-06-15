import os
import pickle as pk

import torch
from core import config
from torch.utils.data import Dataset

from cifar import constants


class Cifar100Dataset(Dataset):
    def __init__(self, imgs, labels, transforms=None):
        imgs = imgs.reshape(
            -1, constants.INPUT_CHANNELS, constants.INPUT_HEIGHT, constants.INPUT_WIDTH
        )
        self.imgs = torch.from_numpy(imgs)

        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transforms = transforms

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.labels)


def unpickle(file_path):
    with open(file_path, "rb") as f:
        dict_data = pk.load(f, encoding="latin1")
    return dict_data


def get_label_mappings(file_path):
    meta = unpickle(file_path)
    label_int_mapping, int_label_mapping = {}, {}
    for i, label in enumerate(meta["fine_label_names"]):
        label_int_mapping[label] = i
        int_label_mapping[i] = label
    return label_int_mapping, int_label_mapping


def load_dataset(
    config: config.DataConfig,
    train: bool = True,
    dry_run: bool = False,
    transforms=None,
):
    """
    Load CIFAR-100 dataset from disk.

    """
    data_path = os.path.join(config.data_path, "train" if train else "test")

    data = unpickle(data_path)

    imgs, labels = data["data"], data["fine_labels"]
    if dry_run:
        imgs = imgs[:500]
        labels = labels[:500]

    return Cifar100Dataset(imgs, labels, transforms)
