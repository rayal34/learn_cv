import os
from dataclasses import dataclass, field
from typing import Any, Optional

from core.train_utils import generate_default_exp_name


@dataclass
class DataConfig:
    root: str
    num_workers: int
    pin_memory: bool

    train_images_filename: Optional[str] = None
    train_labels_filename: Optional[str] = None
    test_images_filename: Optional[str] = None
    test_labels_filename: Optional[str] = None

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


@dataclass
class ExperimentConfig:
    dataset: DataConfig
    training: TrainingConfig
    scheduler: SchedulerConfig
    optimizer: GenericConfig

    model: Any

    name: str = field(default_factory=generate_default_exp_name)
    seed: int = 42
    dry_run: bool = False
    early_stopping: Optional[EarlyStoppingConfig] = None
