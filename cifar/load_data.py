import os
import pickle as pk

import torch
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision.transforms import v2

from cifar import config, constants


class Cifar100Dataset(Dataset):
    def __init__(self, imgs, labels, label_int_mapping, transforms=None):
        self.imgs = torch.from_numpy(imgs)

        int_labels = [label_int_mapping[label] for label in labels]
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
        imgs = imgs[:500]
        labels = labels[:500]

    imgs = imgs.reshape(
        -1, constants.INPUT_CHANNELS, constants.INPUT_HEIGHT, constants.INPUT_WIDTH
    )
    labels = [int_label_mapping[label] for label in labels]

    return Cifar100Dataset(imgs, labels, label_int_mapping, transforms)


def get_train_transforms(config: config.DataAugmentationConfig):
    return [getattr(v2, aug.type)(**aug.params) for aug in config.dataset_augmentations]


def get_general_transforms():
    transforms = [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=constants.MEANS, std=constants.STDS),
    ]
    return transforms


def get_dataloaders(config: config.ExperimentConfig):

    train_transforms = get_train_transforms(config.train_augmentations)
    general_transforms = get_general_transforms()
    all_transforms = train_transforms + general_transforms
    train_data = load_dataset(
        config.dataset,
        train=True,
        dry_run=config.dry_run,
        transforms=v2.Compose(all_transforms),
    )

    test_data = load_dataset(
        config.dataset,
        train=False,
        transforms=v2.Compose(general_transforms),
    )

    if config.train_augmentations.dataloader_augmentations is not None:
        transforms = [
            getattr(v2, aug.type)(**aug.params)
            for aug in config.train_augmentations.dataloader_augmentations
        ]
        dataloader_transforms = v2.RandomChoice(transforms)

        def collate_fn(batch):
            return dataloader_transforms(*default_collate(batch))
    else:
        collate_fn = None

    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=config.training.batch_size,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_data,
        shuffle=False,
        batch_size=config.training.batch_size,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
    )

    return train_dataloader, test_dataloader
