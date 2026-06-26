import torch
from torch.utils.data import DataLoader, default_collate
from torchvision.transforms import v2

from pets import config, constants
from pets.dataset import load_dataset


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
            imgs, boxes, labels = default_collate(batch)
            imgs, labels = dataloader_transforms(imgs, labels)
            return imgs, boxes, labels
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
