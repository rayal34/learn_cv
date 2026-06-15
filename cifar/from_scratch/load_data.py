import torch
from torch.utils.data import DataLoader, default_collate
from torchvision.transforms import v2

from cifar.from_scratch import config
from cifar.utils.dataset import load_dataset

MEANS = [0.5071, 0.4865, 0.4409]
STDS = [0.2673, 0.2564, 0.2762]


def get_train_transforms(config: config.DataAugmentationConfig):
    return [getattr(v2, aug.type)(**aug.params) for aug in config.dataset_augmentations]


def get_post_augmentation_transforms():
    return [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEANS, std=STDS),
    ]


def get_dataloaders(config: config.ExperimentConfig):

    train_transforms = get_train_transforms(config.train_augmentations)

    post_augmentation_transforms = get_post_augmentation_transforms()

    train_data = load_dataset(
        config.dataset,
        train=True,
        dry_run=config.dry_run,
        transforms=v2.Compose(train_transforms + post_augmentation_transforms),
    )

    test_data = load_dataset(
        config.dataset,
        train=False,
        transforms=v2.Compose(post_augmentation_transforms),
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
