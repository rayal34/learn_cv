import os
import pickle as pk

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from utils.augmentation_utils import ZeroOneScale

from cifar import config, constants


class Cifar100Dataset(Dataset):
    def __init__(self, imgs, labels, label_int_mapping, transforms=None):
        self.imgs = torch.tensor(imgs, dtype=torch.float32)

        int_labels = [label_int_mapping[l] for l in labels]
        self.labels = torch.tensor(int_labels, dtype=torch.long)
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
    meta_path = os.path.join(config.data_path, "meta")
    data_path = os.path.join(config.data_path, "train" if train else "test")

    label_int_mapping, int_label_mapping = get_label_mappings(meta_path)
    data = unpickle(data_path)

    imgs, labels = data["data"], data["fine_labels"]
    if dry_run:
        imgs = imgs[:100]
        labels = labels[:100]

    imgs = imgs.reshape(
        -1, constants.INPUT_CHANNELS, constants.INPUT_HEIGHT, constants.INPUT_WIDTH
    )
    labels = [int_label_mapping[label] for label in labels]

    return Cifar100Dataset(imgs, labels, label_int_mapping, transforms)


def get_dataloaders(config: config.ExperimentConfig):

    train_data = load_dataset(
        config.dataset,
        train=True,
        dry_run=config.dry_run,
        transforms=v2.Compose(
            [
                v2.RandomHorizontalFlip(config.data_augmentations.h_flip_prob),
                v2.RandomRotation(degrees=config.data_augmentations.rotate_range),
                v2.RandomCrop(
                    constants.INPUT_HEIGHT,
                    padding=config.data_augmentations.crop_padding,
                    padding_mode="reflect",
                ),
                ZeroOneScale(
                    min_val=constants.MIN_PIXEL_VALUE, max_val=constants.MAX_PIXEL_VALUE
                ),
                v2.Normalize(mean=constants.MEANS, std=constants.STDS),
                v2.RandomErasing(
                    p=config.data_augmentations.random_erasing_p,
                    scale=config.data_augmentations.random_erasing_scale,
                    ratio=config.data_augmentations.random_erasing_ratio,
                    value=config.data_augmentations.random_erasing_value,  # type: ignore
                ),
            ]
        ),
    )

    test_data = load_dataset(
        config.dataset,
        train=False,
        transforms=v2.Compose(
            [
                ZeroOneScale(
                    min_val=constants.MIN_PIXEL_VALUE, max_val=constants.MAX_PIXEL_VALUE
                ),
                v2.Normalize(mean=constants.MEANS, std=constants.STDS),
            ]
        ),
    )

    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=config.training.batch_size,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
    )
    test_dataloader = DataLoader(
        test_data,
        shuffle=False,
        batch_size=config.training.batch_size,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
    )

    return train_dataloader, test_dataloader
