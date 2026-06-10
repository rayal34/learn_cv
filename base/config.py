import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DataConfig:
    root: str
    num_workers: int
    pin_memory: bool

    train_images_filename: str | None = None
    train_labels_filename: str | None = None
    test_images_filename: str | None = None
    test_labels_filename: str | None = None

    data_path: str = field(init=False, default="${.root}/data")
    model_path: str = field(init=False, default="${.root}/models")
    experiment_path: str = field(init=False, default="${.root}/experiments")

    def __post_init__(self):
        self.data_path = os.path.join(self.root, "data")
        self.model_path = os.path.join(self.root, "models")
        self.experiment_path = os.path.join(self.root, "experiments")


@dataclass
class TrainingConfig:
    batch_size: int
    num_epochs: int


@dataclass
class GenericConfig:
    type: str
    params: dict[str, Any]


@dataclass
class SchedulerConfig(GenericConfig):
    update_freq: str = "epoch"


@dataclass
class DataAugmentationConfig:
    dataset_augmentations: list[GenericConfig]
    dataloader_augmentations: Optional[list[GenericConfig]]


@dataclass
class EarlyStoppingConfig:
    patience: int
    min_delta: float
    higher_is_better: bool
