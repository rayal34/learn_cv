import torch
from torch.utils.data import DataLoader, default_collate
from torchvision.transforms import v2

from cifar.fine_tune import config, constants
from cifar.utils.dataset import load_dataset


def get_train_transforms(config: config.DataAugmentationConfig):
    return [getattr(v2, aug.type)(**aug.params) for aug in config.dataset_augmentations]


def get_pre_augmentation_transforms():
    return [v2.Resize((constants.RESNET_INPUT_SIZE, constants.RESNET_INPUT_SIZE))]


def get_post_augmentation_transforms():

    return [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=constants.RESNET_MEANS, std=constants.RESNET_STDS),
    ]


def get_dataloaders(config: config.ExperimentConfig):

    train_transforms = get_train_transforms(config.train_augmentations)

    pre_augmentation_transforms = get_pre_augmentation_transforms()
    post_augmentation_transforms = get_post_augmentation_transforms()

    train_data = load_dataset(
        config.dataset,
        train=True,
        dry_run=config.dry_run,
        transforms=v2.Compose(
            pre_augmentation_transforms
            + train_transforms
            + post_augmentation_transforms
        ),
    )

    test_data = load_dataset(
        config.dataset,
        train=False,
        transforms=v2.Compose(
            pre_augmentation_transforms + post_augmentation_transforms
        ),
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
